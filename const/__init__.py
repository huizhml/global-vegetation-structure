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
    'Tropical & Subtropical Moist Broadleaf Forests',
    'Tropical & Subtropical Dry Broadleaf Forests',
    'Tropical & Subtropical Coniferous Forests',
    'Temperate Broadleaf & Mixed Forests',
    'Temperate Conifer Forests',
    'Boreal Forests/Taiga',
    'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    'Temperate Grasslands, Savannas & Shrublands',
    'Flooded Grasslands & Savannas',
    'Montane Grasslands & Shrublands',
    'Tundra',
    'Mediterranean Forests, Woodlands & Scrub',
    'Deserts & Xeric Shrublands',
    'Mangroves'
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

coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
palette = ['#150b37', '#3b0964', '#61136e', '#85216b', '#a92e5e', '#cc4248', '#e75e2e', '#f78410', '#fcae12', '#f5db4c'] # 0: '#010005',  '#fcffa4'