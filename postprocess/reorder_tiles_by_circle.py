from pathlib import Path

def reorder_tiles_by_circle(tiles_dir: str):
    '''
    Reorder tiles by circle (from center to outer)
    If tiles are not splitted by zone yet run 'bash postprocess/run.sh 1'
    '''
    tiles_dir = Path(tiles_dir).expanduser()
    tiles_files = tiles_dir.glob('*.txt')
    for tiles_file in tiles_files:
        tiles = tiles_file.read_text().splitlines()

        zone = tiles_file.stem
        idx3_list = sorted(set([tile[3] for tile in tiles]))
        idx4_list = sorted(set([tile[4] for tile in tiles]))
        min_l = min(len(idx3_list), len(idx4_list))
        ordered = []
        for i in range(min_l):
            for loc3 in idx3_list[:i+1]:
                tile_name = f'{zone}{loc3}{idx4_list[i]}'
                if tile_name in tiles:
                    ordered.append(tile_name)
            for loc4 in idx4_list[:i]:
                tile_name = f'{zone}{idx3_list[i]}{loc4}'
                if tile_name in tiles:
                    ordered.append(tile_name)
        if len(idx3_list) > len(idx4_list):
            for letter in idx3_list[min_l:]:
                for loc4 in idx4_list:
                    tile_name = f'{zone}{letter}{loc4}'
                    if tile_name in tiles:
                        ordered.append(tile_name)
        else:
            for letter in idx4_list[min_l:]:
                for loc3 in idx3_list:
                    tile_name = f'{zone}{loc3}{letter}'
                    if tile_name in tiles:
                        ordered.append(tile_name)
                        
                        
        assert len(ordered) == len(tiles)
        assert set(ordered) == set(tiles)
        with open(tiles_file, 'w') as f:
            for tile in ordered:
                f.write(tile + '\n')


if __name__ == '__main__':
    tiles_dir = '~/data/gvs/deploy/tiles_by_zone_for_postprocess'
    reorder_tiles_by_circle(tiles_dir)