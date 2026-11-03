import copy
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from kora.metrica_tracking import load_metrica_tracking, parse_metadata, parse_tracking


def metadata_xml():
    """Tiny EPTS match: a substitute takes a slot and the away slot moves."""
    root = ET.fromstring('''<main><Metadata><GlobalConfig><FrameRate>25</FrameRate>
      <ProviderGlobalParameters>
      <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
      <ProviderParameter><Name>first_half_end</Name><Value>2</Value></ProviderParameter>
      <ProviderParameter><Name>second_half_start</Name><Value>3</Value></ProviderParameter>
      <ProviderParameter><Name>second_half_end</Name><Value>4</Value></ProviderParameter>
      </ProviderGlobalParameters></GlobalConfig><Sessions><Session>
      <MatchParameters><Score idLocalTeam="H" idVisitingTeam="A"/></MatchParameters>
      <FieldSize><Width>105</Width><Height>68</Height></FieldSize></Session></Sessions>
      <Teams><Team id="A"/><Team id="H"/></Teams>
      <Players><Player id="h1" teamId="H"/><Player id="a1" teamId="A"/>
      <Player id="h2" teamId="H"/></Players><Devices><Device><Sensors><Sensor>
      <Channels><Channel id="x"><Unit>normalized</Unit></Channel>
      <Channel id="y"><Unit>normalized</Unit></Channel></Channels>
      </Sensor></Sensors></Device></Devices><PlayerChannels/>
      </Metadata><DataFormatSpecifications/></main>''')
    channels = root.find('Metadata/PlayerChannels')
    for player in ('h1', 'a1', 'h2'):
        for axis in ('x', 'y'):
            ET.SubElement(channels, 'PlayerChannel', id=f'{player}_{axis}',
                          playerId=player, channelId=axis)
    for first, last, slots in ((1, 2, ('h1', 'a1')), (3, 4, ('a1', 'h2'))):
        spec = ET.SubElement(root.find('DataFormatSpecifications'),
                             'DataFormatSpecification', startFrame=str(first),
                             endFrame=str(last), separator=':')
        ET.SubElement(spec, 'StringRegister', name='frameCount')
        players = ET.SubElement(spec, 'SplitRegister', separator=';')
        for player in slots:
            pair = ET.SubElement(players, 'SplitRegister', separator=',')
            # Exercise channel lookup rather than assumed x/y order.
            for axis in ('y', 'x'):
                ET.SubElement(pair, 'PlayerChannelRef', playerChannelId=f'{player}_{axis}')
        ball = ET.SubElement(spec, 'SplitRegister', separator=',')
        for axis in ('x', 'y'):
            ET.SubElement(ball, 'BallChannelRef', channelId=axis)
    return root


ROWS = ('1:0,0;1,1:NaN,NaN', '2:NaN,0.5;0.5,NaN:0.5,0.5',
        '3:0.5,0.5;1.1,-0.1:NaN,NaN', '4:0,1;1,0:NaN,NaN')


class MetricaTrackingTests(unittest.TestCase):
    def metadata(self, root=None):
        return parse_metadata(ET.tostring(metadata_xml() if root is None else root))

    def test_substitutions_and_slot_reordering_keep_stable_identity_and_team(self):
        meta = self.metadata()
        original = copy.deepcopy(ROWS)
        match = parse_tracking(ROWS, meta)
        self.assertEqual(match.player_ids, ('h1', 'a1', 'h2'))
        np.testing.assert_array_equal(match.team, [0, 1, 0])
        np.testing.assert_array_equal(match.frame, [1, 2, 3, 4])
        np.testing.assert_array_equal(match.section, [1, 1, 2, 2])
        self.assertEqual(match.halves, ((1, 2), (3, 4)))
        self.assertEqual(match.players.shape, (4, 3, 2))
        self.assertTrue(np.isnan(match.players[:2, 2]).all())
        self.assertTrue(np.isnan(match.players[2:, 0]).all())
        np.testing.assert_allclose(match.players[2, 1], [0, 0])
        np.testing.assert_allclose(match.players[3, 2], [-52.5, 34])
        self.assertEqual(ROWS, original)

    def test_coordinates_preserve_nan_and_off_pitch_values_without_axis_flip(self):
        match = parse_tracking(ROWS, self.metadata())
        np.testing.assert_allclose(match.players[0, 0], [-52.5, -34])
        np.testing.assert_allclose(match.players[0, 1], [52.5, 34])
        np.testing.assert_allclose(match.players[2, 2], [-63, 40.8], rtol=1e-6)
        self.assertEqual(match.players[1, 0, 0], 0)
        self.assertTrue(np.isnan(match.players[1, 0, 1]))
        self.assertTrue(np.isnan(match.players[1, 1, 0]))
        self.assertEqual(match.players.dtype, np.float32)
        for value in (match.players, match.team, match.frame, match.section):
            self.assertFalse(value.flags.writeable)

    def test_invalid_layout_ranges_fail_instead_of_guessing(self):
        for first, last in (('2', '2'), ('0', '2'), ('1', '0'), ('1', '3'),
                            ('1', '1'), ('1.0', '2'), ('1', '5')):
            root = metadata_xml()
            root.find('DataFormatSpecifications')[0].set('startFrame', first)
            root.find('DataFormatSpecifications')[0].set('endFrame', last)
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                self.metadata(root)

    def test_unknown_teams_channels_duplicate_players_and_wrong_units_fail(self):
        changes = (
            ('Metadata/Players/Player', 'teamId', 'unknown'),
            ('Metadata/Players/Player', 'id', 'a1'),
            ('Metadata/PlayerChannels/PlayerChannel', 'playerId', 'missing'),
            ('Metadata/PlayerChannels/PlayerChannel', 'channelId', 'speed'),
            ('DataFormatSpecifications/DataFormatSpecification/SplitRegister/SplitRegister/'
             'PlayerChannelRef', 'playerChannelId', 'missing'),
            ('DataFormatSpecifications/DataFormatSpecification', 'separator', '|'),
        )
        for path, key, value in changes:
            root = metadata_xml()
            root.find(path).set(key, value)
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                self.metadata(root)
        root = metadata_xml()
        root.find('.//Channel/Unit').text = 'metres'
        with self.assertRaises(ValueError):
            self.metadata(root)
        root = metadata_xml()
        pair = root.find('DataFormatSpecifications')[0][1][1]
        for ref, axis in zip(pair, ('y', 'x')):
            ref.set('playerChannelId', f'h1_{axis}')
        with self.assertRaises(ValueError):
            self.metadata(root)

    def test_invalid_periods_pitch_and_framerate_fail(self):
        for path, value in (('.//FrameRate', '30'), ('.//FieldSize/Width', '0'),
                            ('.//FieldSize/Height', 'NaN'),
                            ('.//ProviderParameter/Value', '2')):
            root = metadata_xml()
            root.find(path).text = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.metadata(root)

    def test_missing_duplicate_unsorted_outside_frames_and_bad_rows_fail(self):
        bad_rows = (ROWS[:-1], ROWS + ('5:0,0;0,0:0,0',),
                    (ROWS[0], ROWS[0], *ROWS[2:]), (ROWS[1], ROWS[0], *ROWS[2:]),
                    ('0:0,0;0,0:0,0', *ROWS[1:]),
                    ('1:0,0:0,0', *ROWS[1:]), ('1:inf,0;0,0:0,0', *ROWS[1:]),
                    ('1:bad,0;0,0:0,0', *ROWS[1:]),
                    ('1:0,0;0,0:0', *ROWS[1:]), ('', *ROWS[1:]))
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_tracking(rows, self.metadata())

    def test_pinned_loader_rejects_different_files(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata = Path(folder) / 'metadata.xml'
            tracking = Path(folder) / 'tracking.txt'
            metadata.write_bytes(ET.tostring(metadata_xml()))
            tracking.write_text('\n'.join(ROWS))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_metrica_tracking(metadata, tracking)


if __name__ == '__main__':
    unittest.main()
