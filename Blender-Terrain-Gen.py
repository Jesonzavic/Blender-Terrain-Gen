'''Terrain generation program based on perlin noise to create terrain in blender, written by Jesonzavic October 2019'''

import random

try:
    import numpy
except ImportError:
    print('Get numpy')

from math import *

##perlin noise algorythm along with fade changes, source from http://rosettacode.org/wiki/Perlin_noise#Python
def perlin_noise(p, x, y, z):
    X = int(x) & 255                  # FIND UNIT CUBE THAT
    Y = int(y) & 255                  # CONTAINS POINT.
    Z = int(z) & 255
    x -= int(x)                                # FIND RELATIVE X,Y,Z
    y -= int(y)                                # OF POINT IN CUBE.
    z -= int(z)
    u = fade(x)                                # COMPUTE FADE CURVES
    v = fade(y)                                # FOR EACH OF X,Y,Z.
    w = fade(z)
    A = p[X  ]+Y; AA = p[A]+Z; AB = p[A+1]+Z      # HASH COORDINATES OF
    B = p[X+1]+Y; BA = p[B]+Z; BB = p[B+1]+Z      # THE 8 CUBE CORNERS,

    return lerp(w, lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),  # AND ADD
                                   grad(p[BA  ], x-1, y  , z   )), # BLENDED
                           lerp(u, grad(p[AB  ], x  , y-1, z   ),  # RESULTS
                                   grad(p[BB  ], x-1, y-1, z   ))),# FROM  8
                   lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),  # CORNERS
                                   grad(p[BA+1], x-1, y  , z-1 )), # OF CUBE
                           lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                   grad(p[BB+1], x-1, y-1, z-1 ))))

def fade(t):
    #Kyle McDonald fade function (found at https://www.reddit.com/r/proceduralgeneration/comments/7avjav/ideal_easefade_functions_for_noise/)
    return (-20*t**3 + 70*t**2-84*t**1+35)*t**4

def lerp(t, a, b):
    return a + t * (b - a)

def grad(hash, x, y, z):
    h = hash & 15                      # CONVERT LO 4 BITS OF HASH CODE
    u = x if h<8 else y                # INTO 12 GRADIENT DIRECTIONS.
    v = y if h<4 else (x if h in (12, 14) else z)
    return (u if (h&1) == 0 else -u) + (v if (h&2) == 0 else -v)
##end of perlin noise algorythm

def seeded_random_num(seed, range):
    '''Returns a number that has been created using given seed in a random function'''

    random.seed(seed, version=2)
    num = random.randint(range[0], range[1])
    random.seed(None, version=2)
    return num

def generate_permutation(seed):
    '''Creates a perlin noise permutation (numbers from 0 to 255 in an order) using a given seed,
       outputs a 1d list of size 256'''

    nums = []
    working_seed = seed
    working_size = len(str(seed))

    for i in range(256):
        while True:
            new_num = seeded_random_num(working_seed, (0, 255))
            if not(new_num in nums):
                nums.append(new_num)
                print(i, working_seed)
                break
            working_seed = (working_seed**2) % 10**working_size
            if str(working_seed)[-1] == '0':
                working_seed += 97
            print('not', i, working_seed)
        working_seed = (working_seed**2) % 10**working_size
    return(nums)

def to_image(array):
    '''Colors an array to represent inputed array data, outputs 2d RGB array with the same size as the inputed'''

    try:
        import scipy.misc
    except ImportError:
        print('Get SciPy 1.1.0')
    image_array = numpy.zeros((len(array), len(array[0])), dtype=(float,3))
    median = scipy.median(array)

    for y in range(len(array)):
        for x in range(len(array[0])):
            gradient = 255*array[y][x]
            image_array[y][x] = (int(gradient), int(gradient), int(gradient))

            array_num = array[y][x]

    return image_array

def generate_landscape(seeds, size, frequency, scale_level, iterations, redistribution):
    '''Generates landscape using iterations of Perlin noise and redistribution, outputs 2d array'''

    frequency = frequency/scale_level

    permutations = []
    for i in range(len(seeds)):
        permutation = generate_permutation(seeds[i])
        p = [None] * 512
        for i in range(256):
            p[i+256] = p[i] = permutation[i]
        permutations.append(p)

    num_array = []
    for y in range(int(size[1])):
        array_sgmt = []
        for x in range(int(size[0])):
            height = 0
            for i in range(iterations):
                height += (1/(3**i))*perlin_noise(permutations[i], (x*frequency*2**i)/3.33, (y*frequency*2**i)/3.33, 0)
            height = height**redistribution
            array_sgmt.append(height)
        num_array.append(array_sgmt)

    return num_array

def create_full_normalised_landscape(size, scale_level, iterations, redistribution, seed):
    '''Creates landscape and normalises between 0 and 1, outputs 2d array'''

    map_seeds = []

    map_seeds.append(seeded_random_num(seed, (10**(len(str(seed))), 10**(len(str(seed))+1)-1)))

    for i in range(iterations-1):
        map_seeds.append(seeded_random_num(map_seeds[-1], (10**(len(str(seed))), 10**(len(str(seed))+1)-1)))

    landscape = generate_landscape(map_seeds, map_size, 1, scale_level, iterations, redistribution)

    land_max = numpy.amax(landscape)
    land_min = numpy.amin(landscape)
    normalised_landscape = landscape[0:]

    for y in range(len(normalised_landscape)):
        for x in range(len(normalised_landscape[0])):
            normalised_landscape[y][x] = (normalised_landscape[y][x]-land_min)/(land_max-land_min)

    return normalised_landscape

def save_and_print_array_image(array):
    ''' Will directly turn array into image, show image, and save image to currently open directory,
    note that SciPy 1.1.0 is nessisary as future releaces do not include the 'toimage' function as of writing'''

    try:
        from PIL import Image
    except ImportError:
        print('Get Pillow (PIL)')
    try:
        import scipy.misc
    except ImportError:
        print('Get SciPy 1.1.0')

    image_array = to_image(array)

    img = scipy.misc.toimage(image_array, high=255, low=0, cmin=0, cmax=255)
    img.show()
    img.save(r"hightmap.png")

def define_blender_mesh(array, xy_scale, hight_scale):
    '''Creates varriables neccessary to create a mesh for a Blender object'''

    verticies = []
    for y in range(len(array)):
        for x in range(len(array[0])):
            verticies.append((x*xy_scale, y*xy_scale, array[y][x]*hight_scale))

    faces = []
    for y in range(len(array)-1):
        for x in range(len(array[0])-1):
            self_num = y*len(array)+x
            faces.append((self_num, self_num+1, self_num+len(array)+1, self_num+len(array)))
    return (verticies, faces)

def create_blender_mesh(blender_mesh):
    '''Note: Only use when file is open in blender,
    Creates Object into a Blender scene from given mesh'''

    import bpy, bmesh
    mesh = bpy.data.meshes.new("mesh")
    obj = bpy.data.objects.new("MyObject", mesh)
    obj.location = bpy.context.scene.cursor.location

    mesh.from_pydata(blender_mesh[0],[], blender_mesh[1])
    mesh.update(calc_edges=True)

    bpy.context.collection.objects.link(obj)

print('started')

map_size = (200, 200)
scale_level = 20
iterations = 5
seed = 12345678765432345678987654
hight_scale = 30
xy_scale = .3
redistribution = 1.3

normalised_landscape = create_full_normalised_landscape(map_size, scale_level, iterations, redistribution, seed)

    ### Put landscape into Blender
mesh = define_blender_mesh(normalised_landscape, xy_scale, hight_scale)
create_blender_mesh(mesh)
    ### Put landscape into image to be saved and shown
#save_and_print_array_image(normalised_landscape)
print('finished')
