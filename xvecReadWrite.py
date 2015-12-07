from yael import yael, ynumpy
import numpy as np
import struct

def readXvecs(filename, dim, count, tonumpy=True):
    extension = filename.strip().split('.')[-1]
    if extension == 'bvecs':
        points = yael.fvec_new(dim * count)
        yael.b2fvecs_read(filename, dim, count, points)
        if tonumpy:
            a = yael.fvec_to_numpy(points, (count, dim))
            yael.free(points)
            return a
        else:
           return points
    elif extension == 'fvecs':
        points = yael.fvec_new(dim * count)
        yael.fvecs_read(filename, dim, count, points)
        if tonumpy:
            a = yael.fvec_to_numpy(points, (count, dim))
            yael.free(points)
            return a
        else:
            return points
    elif extension == 'i8vecs':
        file = open(filename, 'r')
        points = np.zeros((count, dim), dtype='float32')
        for i in xrange(count):
            file.read(4)
            points[i,:] = np.fromfile(file, np.int8, dim).astype('float32')
        return points
    elif extension == 'ivecs':
        points = yael.ivec_new(dim * count)
        yael.ivecs_fread(open(filename, 'r'), points, count, dim)
        if tonumpy:
            a = yael.ivec_to_numpy(points, (count, dim))
            yael.free(points)
            return a
        else:
            return points
    else:
        raise Exception('Bad file extension!')

def readXVecsFromOpenedFile(file, dim, count, extension):
    if extension == 'bvecs':
        points = yael.fvec_new(dim * count)
        yael.b2fvecs_fread(file, points, count)
        a = yael.fvec_to_numpy(points, (count, dim))
        yael.free(points)
        return a
    elif extension == 'fvecs':
        points = yael.fvec_new(dim * count)
        yael.fvecs_fread(file, points, count, dim)
        a = yael.fvec_to_numpy(points, (count, dim))
        yael.free(points)
        return a
    else:
        raise Exception('Bad file extension!')


def writeXvecs(points, filename):
    if type(points) is not np.ndarray:
        raise Exception('Convert to numpy before serializing!')
    dim = points.shape[1]
    count = points.shape[0]
    extension = filename.strip().split('.')[-1]
    if extension == 'fvecs':
        points = yael.numpy_to_fvec(points.astype('float32'))
        yael.fvecs_write(filename, dim, count, points)
    elif extension == 'bvecs':
        file = open(filename, 'wb')
        #raise Exception('Writing of bvecs is not implemented yet!')
        dimData = struct.pack('i', dim)
        for i in xrange(count):
            point = points[i,:].astype('uint8')
            file.write(dimData)
            point.tofile(file)
        file.close()
    elif extension == 'i8vecs':
        file = open(filename, 'wb')
        dimData = struct.pack('i', dim)
        for i in xrange(count):
            point = points[i,:].astype('int8')
            file.write(dimData)
            point.tofile(file)
        file.close()
    elif extension == 'ivecs':
        points = yael.numpy_to_ivec(points)
        yael.ivecs_write(filename, dim, count, points)
    else:
        raise Exception('Bad file extension!')

def writeXvecsToOpenFile(points, file, extension):
    if type(points) is not np.ndarray:
        raise Exception('Convert to numpy before serializing!')
    dim = points.shape[1]
    count = points.shape[0]
    if extension == 'fvecs':
        points = yael.numpy_to_fvec(points.astype('float32'))
        yael.fvecs_fwrite(file, dim, count, points)
    elif extension == 'bvecs':
        raise Exception('Writing of bvecs is not implemented yet!')
    elif extension == 'ivecs':
        points = yael.numpy_to_ivec(points)
        yael.ivecs_fwrite(file, dim, count, points)
    else:
        raise Exception('Bad file extension!')




