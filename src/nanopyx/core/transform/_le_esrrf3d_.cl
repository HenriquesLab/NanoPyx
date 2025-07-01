__kernel void interpolate_z_1d(__global float* image, __global float* image_out, float magnification_z, int frame_i) {
  int sM = get_global_id(0);
  int r = get_global_id(1);
  int c = get_global_id(2);
  float slice = sM / magnification_z;

  int slicesM = get_global_size(0);
  int rows = get_global_size(1);
  int cols = get_global_size(2);
  int slices = (int)(slicesM / magnification_z);

  int slice0 = (int)floor(slice);
  int slice1 = slice0 + 1;

  float weight1 = slice - slice0;
  float weight0 = 1.0f - weight1;

  if (magnification_z == 1) {
    image_out[sM * rows * cols + r * cols + c] =
        image[frame_i * slices * rows * cols + slice0 * rows * cols + r * cols + c];
  }
  else if (slice0 >= 0 && slice1 < slices) {
    image_out[sM * rows * cols + r * cols + c] =
        weight0 * image[frame_i * slices * rows * cols + slice0 * rows * cols + r * cols + c] +
        weight1 * image[frame_i * slices * rows * cols + slice1 * rows * cols + r * cols + c];
  } else if (slice0 >= 0) {
    image_out[sM * rows * cols + r * cols + c] =
        image[frame_i * slices * rows * cols + slice0 * rows * cols + r * cols + c];
  } else if (slice1 < slices) {
    image_out[sM * rows * cols + r * cols + c] =
        image[frame_i * slices * rows * cols + slice1 * rows * cols + r * cols + c];
  } else {
    image_out[sM * rows * cols + r * cols + c] = 0.0f;
  }
}

float _c_cubic(float v) {
  float a = 0.5;
  float z = 0;
  if (v < 0) {
    v = -v;
  }
  if (v < 1) {
    z = v * v * (v * (-a + 2) + (a - 3)) + 1;
  } else if (v < 2) {
    z = -a * v * v * v + 5 * a * v * v - 8 * a * v + 4 * a;
  }
  return z;
}

float _c_interpolate(__global float *image, float row, float col, int rows, int cols) {
  int r = (int)row;
  int c = (int)col;
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }
  return image[r * cols + c];
}

float _c_interpolate_cr(__global float *image, float r, float c, int rows, int cols) {
  // return 0 if r OR c positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor((float) (r - 0.5));
  const int c_int = (int)floor((float) (c - 0.5));
  float q = 0;
  float p = 0;

  int r_neighbor, c_neighbor;

  for (int j = 0; j < 4; j++) {
    c_neighbor = c_int - 1 + j;
    p = 0;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }

    for (int i = 0; i < 4; i++) {
      r_neighbor = r_int - 1 + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      p = p + image[r_neighbor * cols + c_neighbor] *
                  _c_cubic(r - (r_neighbor + 0.5));
    }
    q = q + p * _c_cubic(c - (c_neighbor + 0.5));
  }
  return q;
}

__kernel void interpolate_xy_2d(__global float* image, __global float* image_out, float magnification_xy, int frame_i) {
  int s = get_global_id(0);
  int rM = get_global_id(1);
  int cM = get_global_id(2);

  float row = rM / magnification_xy;
  float col = cM / magnification_xy;

  int slices = get_global_size(0);
  int rowsM = get_global_size(1);
  int colsM = get_global_size(2);

  int rows = (int)(rowsM / magnification_xy);
  int cols = (int)(colsM / magnification_xy);

  if (magnification_xy == 1) {
    image_out[s * rowsM * colsM + rM * colsM + cM] =
        image[frame_i * slices * rows * cols + s * rows * cols + rM * cols + cM];
  }
  else {
    image_out[s * rowsM * colsM + rM * colsM + cM] =
        _c_interpolate_cr(&image[frame_i * slices * rows * cols + s * rows * cols], row, col, rows, cols);
  }
}


void _c_gradient_3d(__global float* image, __global float* imGc, __global float* imGr, __global float* imGs, int slices, int rows, int cols, int z_i, int y_i, int x_i) {
  float z0x0y0, z0x0y1, z0x1y0, z0x_1y0, z0x0y_1;
  float z1x0y0, z_1x0y0;

  int z_plus1, y_plus1, x_plus1,
      z_minus1, y_minus1, x_minus1;

  z_plus1 = z_i >= slices - 1 ? slices - 1 : z_i + 1;
  y_plus1 = y_i >= rows - 1 ? rows - 1 : y_i + 1;
  x_plus1 = x_i >= cols - 1 ? cols - 1 : x_i + 1;

  z_minus1 = z_i <= 0 ? 0 : z_i - 1;
  y_minus1 = y_i <= 0 ? 0 : y_i - 1;
  x_minus1 = x_i <= 0 ? 0 : x_i - 1;

  // z=0
  z0x0y0 = image[z_i * rows * cols + y_i * cols + x_i];  // central pixel
  z0x0y1 = image[z_i * rows * cols + y_plus1 * cols + x_i]; // y+1
  z0x1y0 = image[z_i * rows * cols + y_i * cols + x_plus1]; // x+1
  z0x0y_1 = image[z_i * rows * cols + y_minus1 * cols + x_i]; // y-1
  z0x_1y0 = image[z_i * rows * cols + y_i * cols + x_minus1]; // x-1

  // z=1 
  z1x0y0 = image[z_plus1 * rows * cols + y_i * cols + x_i]; // z+1
  
  // z=-1
  z_1x0y0 = image[z_minus1 * rows * cols + y_i * cols + x_i]; // z-1
  
  imGc[z_i * rows* cols + y_i * cols + x_i] = (z0x1y0 - z0x_1y0)/2;
  imGr[z_i * rows* cols + y_i * cols + x_i] = (z0x0y1 - z0x0y_1)/2;
  imGs[z_i * rows* cols + y_i * cols + x_i] = (z0x0y0 - z_1x0y0)/2;

}

__kernel void gradients_3d(__global float* image, __global float* imGs, __global float* imGc, __global float* imGr, int slices, int rows, int cols, int frame_index) {
  int z_i = get_global_id(0);
  int y_i = get_global_id(1);
  int x_i = get_global_id(2);

  if (z_i < slices && y_i < rows && x_i < cols) {
    _c_gradient_3d(&image[frame_index*slices*rows*cols], &imGc[0], &imGr[0], &imGs[0], slices, rows, cols, z_i, y_i, x_i);
  }
}

float _c_calculate_dw3D(float distance, float distance_xy, float distance_z,float tSS, float tSS_z) {
  float D_weight_xy, D_weight_z, D_weight;
  D_weight_xy = (exp((-distance_xy * distance_xy) / tSS));
  D_weight_z = (exp((-distance_z * distance_z) / tSS_z));
  D_weight = distance * (D_weight_xy * D_weight_z);
  return pow(D_weight, 4);
}

double _c_calculate_dw_3d(double distance_xy, double distance_z, double tSS, double tSS_z) {
  double weight_xy = distance_xy * exp((-distance_xy * distance_xy) / tSS);
  double weight_z = distance_z * exp((-distance_z * distance_z) / tSS_z);
  return pow((weight_xy * weight_z), 4);
}

float _c_calculate_dk3D(float Gx, float Gy, float Gz, float dx, float dy, float dz, float distance) {
  float Dk = sqrt((Gy * dz - Gz * dy) * (Gy * dz - Gz * dy) + (Gz * dx - Gx * dz) * (Gz * dx - Gx * dz) + (Gx * dy - Gy * dx) * (Gx * dy - Gy * dx)) / sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
  if (isnan(Dk)) {
    Dk = distance;
  }
  Dk = 1 - Dk / distance;
  return Dk;
}

float _c_calculate_dk_3d(float Gx, float Gy, float Gz, float dx, float dy, float dz, float distance) {
    // Compute the cross product magnitude in 3D
    float cross_magnitude = sqrt(
        pow(Gy * dz - Gz * dy, 2) +
        pow(Gz * dx - Gx * dz, 2) +
        pow(Gx * dy - Gy * dx, 2)
    );

    // Compute the magnitude of the gradient vector
    float gradient_magnitude = sqrt(Gx * Gx + Gy * Gy + Gz * Gz);

    // Compute Dk
    float Dk = cross_magnitude / gradient_magnitude;

    // Handle NaN case
    if (isnan(Dk)) {
        Dk = distance;
    }

    // Normalize Dk
    Dk = 1 - Dk / distance;

    return Dk;
}

float _c_calculate_rgc3D(int xM, int yM, int sliceM, __global float* imIntGx, __global float* imIntGy, __global float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float voxel_ratio, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) {

    float vx, vy, vz, Gx, Gy, Gz, dx, dy, dz, dz_real, distance, distance_xy, distance_z, distanceWeight, GdotR, Dk;

    float xc = (float)(xM) / magnification_xy;
    float yc = (float)(yM) / magnification_xy;
    float zc = (float)(sliceM) / magnification_z;

    float RGC = 0.0f;
    float distanceWeightSum = 0.0f;

    int _start = -(int)(fwhm);
    int _end = (int)(fwhm + 1);

    int _start_z = -(int)(fwhm_z);
    int _end_z = (int)(fwhm_z + 1);
    
    for (int j = _start; j <= _end; j++) {
        vy = yc + j;

        if (0 < vy && vy <= (rowsM/magnification_xy) - 1) {
            for (int i = _start; i <= _end; i++) {
                vx = xc + i;

                if (0 < vx && vx <= (colsM/magnification_xy) - 1) {
                    for (int k = _start_z; k <= _end_z; k++) {
                        vz = zc + k;

                        if (0 < vz && vz <= (slicesM/magnification_z) - 1) {
                            dx = vx - xc;
                            dy = vy - yc;
                            distance_z = vz - zc; 
                            distance = sqrt(dx * dx + dy * dy + distance_z * distance_z);
                            distance_xy = sqrt(dx * dx + dy * dy);
                            
                            if (distance != 0 && distance_xy <= tSO && distance_z <= tSO_z) {
                                int linear_index = (int)(vz * magnification_z) * rowsM * colsM +
                                                   (int)(magnification_xy * vy) * colsM +
                                                   (int)(magnification_xy * vx);

                                Gx = imIntGx[linear_index];
                                Gy = imIntGy[linear_index];
                                Gz = imIntGz[linear_index];

                                // distanceWeight = _c_calculate_dw3D_isotropic(distance, tSS);
                                distanceWeight = _c_calculate_dw3D(distance, distance_xy, distance_z, tSS, tSS_z);
                                //distanceWeight = _c_calculate_dw_3d(distance_xy, distance_z, tSS, tSS_z);

                                // distanceWeight_xy = _c_calculate_dw_xy(distance_xy, tSS); 
                                // distanceWeight_z = _c_calculate_dw_z(distance_z, tSS_z);
                                
                                distanceWeightSum = distanceWeightSum + distanceWeight;
                                // distanceWeightSum_xy += distanceWeight_xy;
                                // distanceWeightSum_z += distanceWeight_z;
                                GdotR = Gx*dx + Gy*dy + Gz*distance_z;

                                if (GdotR < 0) {
                                    Dk = _c_calculate_dk3D(Gx, Gy, Gz, dx, dy, distance_z, distance);
                                    RGC = RGC + (Dk * distanceWeight);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (distanceWeightSum == 0) {
        return 0;
    }

    RGC = RGC / distanceWeightSum;

    if (RGC >= 0 && sensitivity > 1) {
        RGC = pow(RGC, sensitivity);
    } else if (RGC < 0) {
        RGC = 0;
    }

    if (isnan(RGC)) {
        RGC = 0;
    }

    return RGC;
}

__kernel void calculate_rgc3D(__global float* imIntGz, __global float* imIntGy, __global float* imIntGx, __global float* imM, __global float* tmp_slice, int slicesM, int rowsM, int colsM, int magnification_xy, int magnification_z, float ratio_px, float fwhm, float fwhm_z, float tSO, float tSS, float tSO_z, float tSS_z, float sensitivity, int doIntensityWeighting, int f) {
  
  // Index of the current pixel
  int s = get_global_id(0);
  int row = get_global_id(1);
  int col = get_global_id(2);

  // Output image dimensions
  int nPixels_out = slicesM * rowsM * colsM;

  row = row + fwhm*magnification_xy;
  col = col + fwhm*magnification_xy;
  s = s + fwhm_z*magnification_z;

  if (doIntensityWeighting == 1) {
    tmp_slice[s * rowsM * colsM + row * colsM + col] = _c_calculate_rgc3D(col, row, s, &imIntGx[0], &imIntGy[0], &imIntGz[0], colsM, rowsM, slicesM, magnification_xy, magnification_z, ratio_px, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity) * imM[s * rowsM * colsM + row * colsM + col];
  } else {
    tmp_slice[s * rowsM * colsM + row * colsM + col] = _c_calculate_rgc3D(col, row, s, &imIntGx[0], &imIntGy[0], &imIntGz[0], colsM, rowsM, slicesM, magnification_xy, magnification_z, ratio_px, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity);
  }
}

__kernel void time_projection_average(__global float* current_slice, __global float* output, int frame_i) {
  int s = get_global_id(0);
  int row = get_global_id(1);
  int col = get_global_id(2);

  int rows = get_global_size(1);
  int cols = get_global_size(2);

  int current_index = s * rows * cols + row * cols + col;

  // Ensure proper initialization for the first frame
  if (frame_i == 0) {
    output[current_index] = current_slice[current_index] / (frame_i + 1);
  } else {
    // Accumulate the average
    // output[current_index] = output[current_index];
    output[current_index] = output[current_index] + ((current_slice[current_index] - output[current_index]) / (frame_i + 1));
  }
}

__kernel void time_projection_std(
    __global float* current_slice, 
    __global float* mean_output,
    __global float* variance_output,
    int frame_i
) {
    int s = get_global_id(0);
    int row = get_global_id(1);
    int col = get_global_id(2);

    int rows = get_global_size(1);
    int cols = get_global_size(2);

    int current_index = s * rows * cols + row * cols + col;

    // Ensure proper initialization for the first frame
    if (frame_i == 0) {
        mean_output[current_index] = current_slice[current_index];
        variance_output[current_index] = 0.0f;
    } else {
        // Update the running mean
        float delta = current_slice[current_index] - mean_output[current_index];
        mean_output[current_index] += delta / (frame_i + 1);
        float delta2 = current_slice[current_index] - mean_output[current_index];
        // Update the running variance
        variance_output[current_index] += delta * delta2;
    }
}