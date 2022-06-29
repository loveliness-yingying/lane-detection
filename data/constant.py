# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution
"""
tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
"""
tusimple_row_anchor = [ 144,  154,  164,  174,  184,  194,  204,  214,  224, 234, 244, 254, 264,
            274, 284, 294, 304, 314, 324, 334, 344, 354, 364, 374, 384, 394,
            404, 414, 424, 434, 444, 454, 464, 474, 484, 494, 504, 514, 524,
            534, 544, 554, 564, 574, 584, 594, 604, 614, 624, 634, 644, 654,
            664, 674, 684, 694]
#culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

culane_row_anchor = [115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265,
                     275, 285, 295, 305, 315, 325, 335, 345, 355, 365, 375, 385, 395, 405, 415,425, 435, 445, 455, 465,
                     475, 485, 495, 505, 515, 525, 535, 545, 555, 565, 575, 585]