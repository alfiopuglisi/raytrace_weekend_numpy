#
# A Python and numpy implementation of
# https://raytracing.github.io/books/RayTracingInOneWeekend.html 
#
# A. Puglisi, Aug 2020.

import time
import numpy as np
from PIL import Image
np.seterr(invalid='ignore')   # Do not warn about NaNs

class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.z = np.array(z, dtype=np.float32)

    @staticmethod
    def empty(size):
        x = np.empty(size, dtype=np.float32)
        y = np.empty(size, dtype=np.float32)
        z = np.empty(size, dtype=np.float32)
        return Vec3(x,y,z)

    @staticmethod
    def zeros(size):
        x = np.zeros(size, dtype=np.float32)
        y = np.zeros(size, dtype=np.float32)
        z = np.zeros(size, dtype=np.float32)
        return Vec3(x,y,z)

    @staticmethod
    def ones(size):
        x = np.ones(size, dtype=np.float32)
        y = np.ones(size, dtype=np.float32)
        z = np.ones(size, dtype=np.float32)
        return Vec3(x,y,z)
    
    @staticmethod
    def where(condition, v1, v2):
        x = np.where(condition, v1.x, v2.x)
        y = np.where(condition, v1.y, v2.y)
        z = np.where(condition, v1.z, v2.z)
        return Vec3(x,y,z)
    
    def clip(self, vmin, vmax):
        x = np.clip(self.x, vmin, vmax)
        y = np.clip(self.y, vmin, vmax)
        z = np.clip(self.z, vmin, vmax)
        return Vec3(x,y,z)

    def fill(self, value):
        self.x.fill(value)
        self.y.fill(value)
        self.z.fill(value)

    def repeat(self, n):
        x = np.repeat(self.x, n)
        y = np.repeat(self.y, n)
        z = np.repeat(self.z, n)
        return Vec3(x,y,z)
    
    def __str__(self):
        return 'vec3: x:%s y:%s z:%s' % (str(self.x), str(self.y), str(self.z))
    
    def __len__(self):
        return self.x.size

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar):
        return Vec3(self.x*scalar, self.y*scalar, self.z*scalar)

    def multiply(self, other):
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    def __truediv__(self, scalar):
        return Vec3(self.x/scalar, self.y/scalar, self.z/scalar)
    
    def tile(self, shape):
        '''Replicate np.tile on each component'''
        return Vec3(np.tile(self.x, shape), np.tile(self.y, shape), np.tile(self.z, shape))

    def __getitem__(self, idx):
        '''Extract a vector subset'''
        return Vec3(self.x[idx], self.y[idx], self.z[idx])
    
    def __setitem__(self, idx, other):
        '''Set a vector subset from another vector'''
        self.x[idx] = other.x
        self.y[idx] = other.y
        self.z[idx] = other.z

    def join(self):
        '''Join the three components into a single 3xN array'''
        return np.vstack((self.x, self.y, self.z))
    
    def append(self, other):
        '''Append another vector to this one.
        Use concatenate() because cupy has no append function.
        '''
        self.x = np.concatenate((self.x, other.x))
        self.y = np.concatenate((self.y, other.y))
        self.z = np.concatenate((self.z, other.z))
        

## Aliases
Point3 = Vec3
Color = Vec3

## Utility functions
def unit_vector(v):
    return v / length(v)

def dot(a, b):
    return a.x*b.x + a.y*b.y + a.z*b.z

def length(v):
    return length_squared(v)**0.5

def length_squared(v):
    return v.x*v.x + v.y*v.y + v.z*v.z

def cross(a, b):
    return Vec3(a.y*b.z - a.z*b.y,
                -(a.x*b.z - a.z*b.x),
                a.x*b.y - a.y*b.x)


def convert_to_pil(v, width, height, scale = 255.999):
    
    img = (v.join() * scale).astype(np.uint8)

    if len(img.shape) == 2:
        img_rgb = img.swapaxes(0,1).reshape(height, width, 3)
    else:
        img_rgb = img.reshape(height, width)

    return Image.fromarray(img_rgb)
    

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + self.direction*t
    
    def __getitem__(self, idx):
        return Ray(self.origin[idx], self.direction[idx])

    def __setitem__(self, idx, other):
        self.origin[idx] = other.origin
        self.direction[idx] = other.direction
    
    def __len__(self):
        return self.origin.x.size


class Timer:

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        t1 = time.time()
        print('Elapsed time: %.3f s' % (t1 - self.t0))


def my_random(low, high, size):
    return np.random.uniform(low, high, size).astype(np.float32)


def random_in_unit_sphere(n):
    '''Generate random Vec3 arrays in batches and keep the ones inside the unit sphere'''

    values = Vec3.zeros(0)

    while len(values) < n:
        random_values = Vec3(my_random(-1.0, 1.0, n), my_random(-1.0, 1.0, n), my_random(-1.0, 1.0, n))
        good_ones = length_squared(random_values) < 1
        values.append(random_values[good_ones])
        
    return values[np.arange(n)]


def random_unit_vectors(n):
    a = my_random(0.0, 2.0*np.pi, n)
    z = my_random(-1.0, 1.0, n)
    r = np.sqrt(1 - z*z)
    return Vec3(r*np.cos(a), r*np.sin(a), z)


def random_in_unit_disk(n):
    ''''Generate random Vec3 arrays in batches and keep the ones inside the unit disk'''

    values = Vec3.zeros(0)

    while len(values) < n:
        random_values = Vec3(my_random(-1.0, 1.0, n), my_random(-1.0, 1.0, n), np.zeros(n))
        good_ones = length_squared(random_values) < 1
        values.append(random_values[good_ones])
    
    return values[:n]

   
def render_image(width, height):

    print('%dx%d pixels, %d samples per pixel' % (width, height, samples_per_pixel))

    with Timer():
        ii, jj = np.mgrid[:float(height), :float(width)]

        u = (jj/(width-1)).flatten().astype(np.float32)
        v = (ii/(height-1)).flatten().astype(np.float32)

        cam = get_camera()

        img = Vec3.zeros(width * height)
        for s in range(samples_per_pixel):
            print('Starting sample %d of %d' % ((s+1), samples_per_pixel))

            uu = u + my_random(0.0, 1.0, u.size) / (width - 1)
            vv = v + my_random(0.0, 1.0, v.size) / (height - 1)

            r = cam.get_ray(uu,vv)

            img += ray_color(r)

        img *= 1.0 / samples_per_pixel
        img.x = np.sqrt(img.x)
        img.y = np.sqrt(img.y)
        img.z = np.sqrt(img.z)
        return img.clip(0.0, 0.999)


from collections import namedtuple
from abc import abstractmethod

                           
class HitRecord:
    def __init__(self, n, empty=False):
        self.p           = empty or Vec3.empty(n)
        self.normal      = empty or Vec3.empty(n)
        self.t           = empty or np.full(n, np.inf, dtype=np.float32)
        self.front_face  = empty or np.zeros(n, dtype=np.float32)
        self.index       = empty or np.arange(n, dtype=np.int32)
        self.material_id = empty or np.zeros(n, dtype=np.int64)

    def __getitem__(self, idx):
        other = HitRecord(len(idx), empty=True)
        other.p           = self.p[idx]
        other.normal      = self.normal[idx]
        other.t           = self.t[idx]
        other.front_face  = self.front_face[idx]
        other.index       = self.index[idx]
        other.material_id = self.material_id[idx]
        return other


class Hittable:
    @abstractmethod
    def update_hit_record(rays, t_min, t_max, hit_record: HitRecord):
        pass


class Sphere(Hittable):
    '''A hittable sphere that knows how to update the hit record'''

    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material                 # Changed here

    def update_hit_record(self, rays, t_min, t_max, hit_record):

        oc = rays.origin - self.center
        a = length_squared(rays.direction)
        half_b = dot(oc, rays.direction)
        c = length_squared(oc) - self.radius*self.radius
        discriminant = half_b*half_b - a*c
        
        root = np.sqrt(discriminant)
        t1 = (-half_b - root) / a
        t2 = (-half_b + root) / a
        hit1 = np.logical_and(t1 < t_max, t1 > t_min)
        hit2 = np.logical_and(t2 < t_max, t2 > t_min)
        
        # Combine the two hits, precedence to t1 (closest)
        t = np.where(hit2, t2, np.inf)
        t = np.where(hit1, t1, t)       
        
        # Detect where in the rays list we are the closest hit
        closest = np.where(t < hit_record.t)
        
        # Early exit if nothing hit
        if len(closest[0]) == 0:
            return
        
        # Calculate normal
        hit_rays = rays[closest]
        
        p = hit_rays.at(t[closest])
        outward_normal = (p - self.center) / self.radius 
        front_face = dot(hit_rays.direction, outward_normal) < 0 
        normal = Vec3.where(front_face, outward_normal, -outward_normal)
        
        # Update hit record
        hit_record.p[closest] = p
        hit_record.normal[closest] = normal
        hit_record.t[closest] = t[closest]
        hit_record.front_face[closest] = front_face
        hit_record.material_id[closest] = id(self.material)                 # Changed here




ScatterResult = namedtuple('ScatterResult', 'attenuation rays is_scattered')

def reflect(v, n):
    return v - n*2*dot(v, n)

def refract(uv, n, etai_over_etat):
    
    cos_theta = dot(-uv, n)
    r_out_perp = (uv + n*cos_theta) * etai_over_etat
    r_out_parallel = n * (-np.sqrt(np.abs(1.0 - length_squared(r_out_perp))))
    return r_out_perp + r_out_parallel

def schlick(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * (1 - cosine)**5



class Material:
    def scatter(r_in: Ray, ray_idx, rec: HitRecord) -> ScatterResult:
        pass


class Lambertian(Material):
    
    def __init__(self, albedo: Color):
        self.albedo = albedo
    
    def scatter(self, r_in: Ray, rec: HitRecord) -> ScatterResult:

        scatter_direction = rec.normal + random_unit_vectors(len(r_in))
        scattered = Ray(rec.p, scatter_direction)
        
        return ScatterResult(attenuation = self.albedo,
                             rays = scattered,
                             is_scattered = np.full(len(r_in), True, dtype=np.bool))


class Metal(Material):

    def __init__(self, albedo: Color, f):
        self.albedo = albedo
        self.fuzz = f if f < 1 else 1
        
    def scatter(self, r_in: Ray, rec: HitRecord) -> ScatterResult:

        reflected = reflect(unit_vector(r_in.direction), rec.normal)
        scattered = Ray(rec.p, reflected + random_in_unit_sphere(len(r_in))*self.fuzz)

        return ScatterResult(attenuation = self.albedo,
                             rays = scattered,
                             is_scattered = dot(scattered.direction, rec.normal) > 0)    


class Dielectric(Material):
    
    def __init__(self, ref_idx):
        self.ref_idx = ref_idx

    def scatter(self, r_in: Ray, rec: HitRecord) -> ScatterResult:
        
        etai_over_etat = np.where(rec.front_face, 1.0 / self.ref_idx, self.ref_idx)
        
        unit_direction = unit_vector(r_in.direction)
        
        ## Reflection/refraction choice: calculate both and choose later
        cos_theta = np.fmin(dot(-unit_direction, rec.normal), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta*cos_theta)
        reflected = reflect(unit_direction, rec.normal)
        refracted = refract(unit_direction, rec.normal, etai_over_etat)

        reflected_rays = Ray(rec.p, reflected)
        refracted_rays = Ray(rec.p, refracted)

        reflect_prob = schlick(cos_theta, etai_over_etat)
        random_floats = my_random(0.0, 1.0, len(reflect_prob))

        must_reflect = (etai_over_etat * sin_theta > 1.0)
        again_reflect = (random_floats < reflect_prob)

        all_reflect = np.where(np.logical_or(must_reflect, again_reflect))
        
        refracted_rays[all_reflect] = reflected_rays[all_reflect]
        
        return ScatterResult(attenuation = Color(1.0, 1.0, 1.0),
                             rays = refracted_rays,
                             is_scattered = np.full(len(r_in), True, dtype=np.bool))  


class Camera:

    def __init__(self, lookfrom: Vec3,
                       lookat: Vec3,
                       vup: Vec3,
                       vfov: float,           # vertical field-of-view in degrees
                       aspect_ratio: float,
                       aperture: float,
                       focus_dist: float):

        theta = np.deg2rad(vfov)
        h = np.tan(theta/2)
        viewport_height = 2.0 * h;
        viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat)
        u = cross(vup, w)
        v = -cross(w, u)           # Minus here, to have things looking upright

        self.origin = lookfrom
        self.horizontal = u * viewport_width * focus_dist
        self.vertical = v * viewport_height * focus_dist
        self.lower_left_corner = self.origin - self.horizontal/2 - self.vertical/2 - w*focus_dist
        self.lens_radius = aperture / 2
        self.u = u
        self.v = v

    def get_ray(self, s, t):
        all_origins = self.origin.tile((s.size,))
        rd = random_in_unit_disk(s.size) * self.lens_radius
        offset = self.u * rd.x + self.v * rd.y

        return Ray(all_origins + offset, self.lower_left_corner
                                         + self.horizontal * s
                                         + self.vertical * t
                                         - all_origins - offset)
    

def ray_color(rays):
    '''Iterative version with materials'''

    frame_intensity = Vec3.ones(len(rays))
    frame_rays = rays
    hit_record = HitRecord(len(rays))

    materials = set([x.material for x in world])

    for d in range(max_depth):

        print('Depth %d, %d rays' % (d, len(rays)))
        # Initialize all distances to infinite and propagate all rays
        hit_record.t.fill(np.inf)
        hit_record.material_id.fill(0)
        for hittable in world:
            hittable.update_hit_record(rays, 0.001, np.inf, hit_record)         
            
        for material in materials:

            material_hits = np.where(hit_record.material_id == id(material))[0]
            if len(material_hits) == 0:
                continue

            my_rays = rays[material_hits]
            my_rec = hit_record[material_hits]
            result = material.scatter(my_rays, my_rec)         
            
            # All rays have done something
            rays[material_hits] = result.rays
            frame_rays.direction[my_rec.index] = result.rays.direction
                
            intensity = result.attenuation.multiply(frame_intensity[my_rec.index])
            intensity[np.where(~result.is_scattered)] = Vec3(0,0,0)

            frame_intensity[my_rec.index] = intensity

            # Those that have been scattered stop here
            not_scattered_material_idx = material_hits[~result.is_scattered]
            hit_record.t[not_scattered_material_idx] = np.inf
            
        # Iterate with those rays that have been scattered by something
        scattered_rays = np.where(hit_record.t != np.inf)[0]
        rays = rays[scattered_rays]
        hit_record = hit_record[scattered_rays]

        if len(rays) == 0:
            break
            
    unit_direction = unit_vector(frame_rays.direction)
    t = 0.5 * unit_direction.y + 1.0
    img = (Color(1.0, 1.0, 1.0) * (1 - t) + Color(0.5, 0.7, 1.0) * t).multiply(frame_intensity)

    return img



def render(width, height):
    colors = render_image(width, height)
    image = convert_to_pil(colors, width, height)
    return image


def random_double(low=0.0, high=1.0):
    return np.random.uniform(low, high, 1).astype(np.float32)

def vec3_random(low=0.0, high=1.0):
    r = my_random(low, high, size=3)
    return Vec3(r[0], r[1], r[2])


ground_material = Lambertian(Color(0.5, 0.5, 0.5))

world = []
world.append(Sphere(Point3(0, -1000, 0), 1000, ground_material))

for a in range(-11, 11):
    for b in range(-11, 11):
        choose_mat = random_double()
        center = Point3(a + 0.9*random_double(), 0.2, b + 0.9*random_double())

        if length(center - Point3(4, 0.2, 0)) > 0.9:

            if choose_mat < 0.8:
                ## diffuse
                albedo = vec3_random().multiply(vec3_random())
                sphere_material = Lambertian(albedo)
                world.append(Sphere(center, 0.2, sphere_material))

            elif choose_mat < 0.95:
                ## metal
                albedo = vec3_random(0.5, 1)
                fuzz = random_double(0, 0.5)
                sphere_material = Metal(albedo, fuzz)
                world.append(Sphere(center, 0.2, sphere_material))

            else:
                ## glass
                sphere_material = Dielectric(1.5)
                world.append(Sphere(center, 0.2, sphere_material))


world.append(Sphere(Point3( 0, 1, 0), 1.0, Dielectric(1.5)))
world.append(Sphere(Point3(-4, 1, 0), 1.0, Lambertian(Color(0.4, 0.2, 0.1))))
world.append(Sphere(Point3( 4, 1, 0), 1.0, Metal(Color(0.7, 0.6, 0.5), 0.0)))

image_width = 1200
aspect_ratio = 16/9
image_height = int(image_width / aspect_ratio)
samples_per_pixel = 10
max_depth = 50

def get_camera():
    lookfrom = Point3(13,2,3)
    lookat = Point3(0,0,0)
    
    return Camera(lookfrom = lookfrom,
                  lookat   = lookat,
                  vup      = Vec3(0,1,0),
                  vfov     = 20,
                  aspect_ratio = aspect_ratio,
                  aperture     = 0.1,
                  focus_dist   = 10.0)


img = render(image_width, image_height)
img.save('img.bmp')
