import os
import h5py
import numpy as np
import zarr
import gdal


def init_h5_output(prediction_fp, compression, comp_level, chunk_size, img_width, img_height):
    """Initialize output as a HDF5 file using h5py for continuous patch prediction writing
    The output is a 3D array with shape (101, img_width, img_height)
    NOTE: It's very slow in writing, so it's not used in the current implementation
    Args:
        prediction_fp: path to the prediction file
        compression: compression method
        comp_level: compression level
        chunk_size: chunk size for spatial dimension
        img_width: width of the image
        img_height: height of the image
    """
    
    prediction_fp = prediction_fp.with_suffix('.h5')
    if prediction_fp.exists():
        os.remove(prediction_fp)
    config = dict(
            compression=compression, compression_opts=comp_level,
            dtype=np.int16,
            chunks=(101, chunk_size, chunk_size)
        )
    h5_output = h5py.File(prediction_fp, 'w')
    dset = h5_output.create_dataset('predictions', shape=(101, img_width, img_height), **config)
    return dset
     
def write_patch_predictions_h5(dset, prediction, y_topleft, x_topleft, patch_size_no_border):
    """Windowed write patch predictions directly to HDF5
    Args:
        dset: the dataset to write to
        prediction: the prediction to write
        y_topleft: the y coordinate of the top left corner of the patch
        x_topleft: the x coordinate of the top left corner of the patch
        patch_size_no_border: the size of the patch without border
    """
    dset[:, y_topleft:y_topleft + patch_size_no_border, x_topleft:x_topleft + patch_size_no_border] = prediction
    
    
def init_zipzarr_output(prediction_fp, compression, comp_level, chunk_size, img_width, img_height):
    """Initialize output as a Zip file using Zarr
    NOTE: Not used because we want to host the data on MPC and visualize it on the web explorer
    Args:
        prediction_fp: path to the prediction file
        compression: compression method
        comp_level: compression level
        chunk_size: chunk size for spatial dimension
        img_width: width of the image
        img_height: height of the image
    """
    prediction_fp = prediction_fp.with_suffix('.zip')
    if prediction_fp.exists():
        os.remove(prediction_fp)
    config = dict(  
        compressors=zarr.codecs.BloscCodec(cname=compression, clevel=comp_level, shuffle=zarr.codecs.BloscShuffle.bitshuffle),
        dtype=np.int16,
        chunks=(101, chunk_size, chunk_size)
    )
    store = zarr.storage.ZipStore(prediction_fp, mode='w')
    zip_output = zarr.create_array(store=store, shape=(101, img_width, img_height), **config)
    return zip_output


def write_patch_predictions_zip(zip_output, prediction, y_topleft, x_topleft, patch_size_no_border):
    """Write patch predictions directly to Zip using Zarr
    Args:
        zip_output: the zip output to write to
        prediction: the prediction to write
        y_topleft: the y coordinate of the top left corner of the patch
        x_topleft: the x coordinate of the top left corner of the patch
        patch_size_no_border: the size of the patch without border
    """
    prediction_no_border = prediction[::3]
    zip_output[:, y_topleft:y_topleft + patch_size_no_border, x_topleft:x_topleft + patch_size_no_border] = prediction_no_border
    
    
def init_gtiff_output(prediction_fp, compression, comp_level, chunk_size, img_width, img_height):
    """Initialize output as a Cloud-Optimized GeoTIFF using GDAL directly
    Args:
        prediction_fp: path to the prediction file
        compression: compression method
        comp_level: compression level
        chunk_size: chunk size for spatial dimension
        img_width: width of the image
        img_height: height of the image
    """
    prediction_fp = prediction_fp.with_suffix('.tif')
    if prediction_fp.exists():
        os.remove(prediction_fp)
    
    gdal.UseExceptions()
    # Get number of bands
    n_bands = 101
        
    # Set creation options
    options = [
        'TILED=YES',
        'BLOCKXSIZE={}'.format(chunk_size), # a different block size than the actual patch size is slow
        'BLOCKYSIZE={}'.format(chunk_size),
        'PREDICTOR=2',
        'BIGTIFF=YES',
        'NUM_THREADS=ALL_CPUS'
        
    ]
    if compression == 'WEBP':
        options.append('COMPRESS={}'.format(compression))
        options.append('QUALITY={}'.format(comp_level))
        options.append('INTERLEAVE=PIXEL')
        
    elif compression is None:
        options.append('COMPRESS=None')
        options.append('INTERLEAVE=BAND')
    else:
        options.append('COMPRESS={}'.format(compression))
        options.append('ZLEVEL={}'.format(comp_level))
        options.append('INTERLEAVE=BAND')
    
    # Create the dataset directly
    driver = gdal.GetDriverByName('GTiff')
    tiff_output = driver.Create(
        str(prediction_fp),
        xsize=img_width,
        ysize=img_height,
        bands=n_bands,
        eType=gdal.GDT_Int16,
        options=options
    )
    return tiff_output


def write_patch_predictions_gtiff(tiff_output, prediction, y_topleft, x_topleft, patch_size_no_border):
    """Write patch predictions directly to COG using GDAL
    Args:
        tiff_output: the tiff output to write to
        prediction: the prediction to write
        y_topleft: the y coordinate of the top left corner of the patch
        x_topleft: the x coordinate of the top left corner of the patch
        patch_size_no_border: the size of the patch without border
    """
    prediction_no_border = prediction[::3]
    if prediction_no_border is None:
        return
    
    # Write each band
    for i in range(prediction_no_border.shape[0]):
        band = tiff_output.GetRasterBand(i + 1)
        band.WriteArray(
            prediction_no_border[i],
            xoff=x_topleft,
            yoff=y_topleft
        )
    
    # Flush cache
    tiff_output.FlushCache()