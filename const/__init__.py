NO_DATA = 32767
# From train*_filtered_v1
LAT_MEAN = 12.7596
LAT_STD = 25.6075
LON_SIN_MEAN = 0.1098
LON_SIN_STD = 0.7536
LON_COS_MEAN = 0.3072
LON_COS_STD = 0.5706
SLOPE_MEAN = 6.5781
SLOPE_STD = 8.9007

SCL_EXCLUDE_LABELS = [0, 1, 3, 8, 9, 10, 11, 65535]
SCL_WATER = 6
# predicted land cover labels
ESA_SNOW = 7 # = 70 in ESA World Cover
ESA_BUILT_UP = 5 # = 50 in ESA World Cover
ESA_WATER = 8 # = 80 in ESA World Cover

# ESA World cover original labels
ESA_SNOW_RAW = 70
ESA_BUILT_UP_RAW = 50
ESA_WATER_RAW = 80
ESA_UNKNOWN_RAW = 0

ESA_DATETIME = '2021-01-01/2021-12-31'
ESA_WC = {
    'unknown': 0,
    'Tree cover': 10,
    'Shrubland': 20,
    'Grassland': 30,
    'Cropland': 40,
    'Built-up': 50,
    'Bare / sparse vegetation': 60,
    'Snow and ice': 70,
    'Permanent water bodies': 80,
    'Herbaceous wetland': 90,
    'Mangroves': 95,
    'Moss and lichen': 100,
}

ESA_WC_s = {
    0:'unknown',
    1: 'Tree',
    2: 'Shrub',
    3: 'Grass',
    4: 'Crop',
    5: 'Built',
    6: 'Bare',
    7: 'Snow',
    8: 'Water',
    9: 'Herb',
    10: 'Moss', 
    11: 'Mangroves'
}

BIOMES = [
    # Don't change the order, index is the corresponding BIOME number
    {'value': 1, 'abbr': 'Tro.Sub.Moi.Br.F', 'name': 'Tropical & Subtropical Moist Broadleaf Forests', }, # 1
    {'value': 2, 'abbr': 'Tro.Sub.Dry.Br.F', 'name': 'Tropical & Subtropical Dry Broadleaf Forests'},   # 2
    {'value': 3, 'abbr': 'Tro.Sub.Con.F', 'name': 'Tropical & Subtropical Coniferous Forests'},     # 3
    {'value': 4, 'abbr': 'Tem.Br.Mix.F', 'name': 'Temperate Broadleaf & Mixed Forests'},          # 4
    {'value': 5, 'abbr': 'Tem.Con.F', 'name': 'Temperate Conifer Forests'},                     # 5
    {'value': 6, 'abbr': 'Bor.F.Tai', 'name': 'Boreal Forests/Taiga'},                          # 6
    {'value': 7, 'abbr': 'Tro.Sub.Gr.Sav.Shr', 'name': 'Tropical & Subtropical Grasslands, Savannas & Shrublands'}, # 7
    {'value': 8, 'abbr': 'Tem.Gr.Sav.Shr', 'name': 'Temperate Grasslands, Savannas & Shrublands'},          # 8
    {'value': 9, 'abbr': 'Flo.Gr.Sav', 'name': 'Flooded Grasslands & Savannas'},                        # 9
    {'value': 10, 'abbr': 'Mon.Gr.Sh', 'name': 'Montane Grasslands & Shrublands'},                    # 10
    {'value': 11, 'abbr': 'Tun', 'name': 'Tundra'},                                            # 11
    {'value': 12, 'abbr': 'Med.F.Woo.Scr', 'name': 'Mediterranean Forests, Woodlands & Scrub'},          # 12
    {'value': 13, 'abbr': 'Des.Xer.Shr', 'name': 'Deserts & Xeric Shrublands'},                          # 13
    {'value': 14, 'abbr': 'Man', 'name': 'Mangroves'},                                    # 14
    {'value': 98, 'abbr': 'biome_98', 'name': 'Biome 98'},                                    # 98
    {'value': 99, 'abbr': 'biome_99', 'name': 'Biome 99'}                                    # 99
]

BIOMES_s = {
    1: 'Tropical Moist Broadleaf',
    2: 'Tropical Dry Broadleaf',
    3: 'Tropical Coniferous',
    4: 'Temperate Broadleaf',
    5: 'Temperate Conifer',
    6: 'Boreal',
    7: 'Tropical Grasslands',
    8: 'Temperate Grasslands',
    9: 'Flooded Grasslands',
    10: 'Montane Grasslands',
    11: 'Tundra',
    12: 'Mediterranean Forests',
    13: 'Deserts',
    14: 'Mangroves'
}

MASKED_VALUE = 32767
RH100_INDEX = 301
RH98_INDEX = 295
KEY_RHS = (0, 10, 25, 50, 75, 95, 98, 100)
CHM_COLS=['rh98', 'rh95', 'rh100', 'lc', 'slope', 'lat', 'lon', 'shot_number', 'RH95_UMD', 'RH98_ETH', 'RH100_UM', 'RH95_META', 'RH95_Q1_raw', 'RH98_Q1_raw', 'RH100_Q1_raw']
LUMI_PROJECT=465002698

coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
palette = ['#150b37', '#3b0964', '#61136e', '#85216b', '#a92e5e', '#cc4248', '#e75e2e', '#f78410', '#fcae12', '#f5db4c'] # 0: '#010005',  '#fcffa4'
rh_vis_params = {
    'RH25': {
        'cmin': 0,
        'cmax': 120,
    },
    'RH50': {
        'cmin': 0,
        'cmax': 200,
    },
    'RH75': {
        'cmin': 0,
        'cmax': 300,
    },
    'RH98': {
        'cmin': 0,
        'cmax': 500,
    }
}