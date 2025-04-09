__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void
conv2d(__read_only image2d_t image_in, __write_only image2d_t image_out, __read_only image2d_t kernel_array) {

  const int2 image_size = get_image_dim(image_in);
  const int2 kernel_size = get_image_dim(kernel_array);
  int2 image_coords = (int2)(get_global_id(1), get_global_id(2));
  int2 kernel_center = (kernel_size-1)/2;

  float acc = 0;
  int2 local_coords;
  int2 kernel_coords;

  for (int kr = 0; kr<kernel_size.x; kr++) {
    for (int kc = 0; kc<kernel_size.y; kc++) {

      kernel_coords = (int2)(kr,kc);
      local_coords = image_coords + (kernel_coords-kernel_center);

      acc = acc + read_imagef(kernel_array,sampler,kernel_coords).x * read_imagef(image_in,sampler,local_coords).x;
      }
    }

  write_imagef(image_out,image_coords,acc);
}

__kernel void
conv2d_2(__global float *image, __global float *image_out, __global float *kernel_array, int kernel_size){

  int frame = get_global_id(0);
  int row = get_global_id(1);
  int col = get_global_id(2);

  int nframes = get_global_size(0);
  int nrows = get_global_size(1);
  int ncols = get_global_size(2);

  int kernel_center = (kernel_size-1)/2;

  float acc = 0;
  int localrow;
  int localcol;

  for (int kr = 0; kr<kernel_size; kr++) {
    for (int kc = 0; kc<kernel_size; kc++) {

      // localrow and localcol should lie between 0 and imgsize-1
      localrow = min(nrows-1,max(0,row + (kr-kernel_center)));
      localcol = min(ncols-1,max(0,col + (kc-kernel_center)));



      acc = acc + kernel_array[kr*kernel_size+kc] * image[frame*nrows*ncols+localrow*ncols+localcol];
      }
    }

image_out[frame*nrows*ncols+row*ncols+col] = acc;

}