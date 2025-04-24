float catmull_rom_weight(float t, int offset) {
  // Catmull-Rom spline weight calculation
  t = fabs(t - offset);
  if (t < 1.0f) {
    return 1.0f - 2.0f * t * t + t * t * t;
  } else if (t < 2.0f) {
    return 4.0f - 8.0f * t + 5.0f * t * t - t * t * t;
  }
  return 0.0f;
}


__kernel void interpolate_3d(__global float* image, __global float* magnified_image, int magnification_xy, int magnification_z, int f) {
  int s = get_global_id(0);
  int row = get_global_id(1);
  int col = get_global_id(2);

  int slices = get_global_size(0) / magnification_z;
  int rows = get_global_size(1) / magnification_xy;
  int cols = get_global_size(2) / magnification_xy;

  // Linear interpolation in z
  float z_ratio = (float)s / magnification_z;
  int z0 = (int)floor(z_ratio);
  int z1 = z0 + 1;
  z1 = z1 < slices ? z1 : slices - 1;
  float z_weight = z_ratio - z0;

  // Bicubic interpolation in xy
  float y_ratio = (float)row / magnification_xy;
  float x_ratio = (float)col / magnification_xy;
  int y0 = (int)floor(y_ratio);
  int x0 = (int)floor(x_ratio);
  float y_weight = y_ratio - y0;
  float x_weight = x_ratio - x0;

  float result = 0.0f;
  for (int dz = 0; dz <= 1; dz++) {
    int z = dz == 0 ? z0 : z1;
    float z_contrib = dz == 0 ? (1 - z_weight) : z_weight;

    for (int dy = -1; dy <= 2; dy++) {
      int y = y0 + dy;
      y = y > 0 ? (y < rows ? y : rows - 1) : 0;
      float y_contrib = catmull_rom_weight(y_weight, dy);

      for (int dx = -1; dx <= 2; dx++) {
        int x = x0 + dx;
        x = x > 0 ? (x < cols ? x : cols - 1) : 0;
        float x_contrib = catmull_rom_weight(x_weight, dx);

        float pixel_value = image[f * slices * rows * cols + z * rows * cols + y * cols + x];
        result += pixel_value * z_contrib * y_contrib * x_contrib;
      }
    }
  }

  magnified_image[s * get_global_size(1) * get_global_size(2) + row * get_global_size(2) + col] = result;
}

void _c_gradient_3d(__global float* image, __global float* imGc, __global float* imGr, __global float* imGs, int slices, int rows, int cols, int z_i, int y_i, int x_i) {
  float ip0, ip1, ip2, ip3, ip4, ip5, ip6, ip7;

  int z_1, y_1, x_1;

  z_1 = z_i >= slices - 1 ? slices - 1 : z_i + 1;
  y_1 = y_i >= rows - 1 ? rows - 1 : y_i + 1;
  x_1 = x_i >= cols - 1 ? cols - 1 : x_i + 1;
  ip0 = image[z_i * rows * cols + y_i * cols + x_i];
  ip1 = image[z_i * rows * cols + y_i * cols + x_1];
  ip2 = image[z_i * rows * cols + y_1 * cols + x_i];
  ip3 = image[z_i * rows * cols + y_1 * cols + x_1];
  ip4 = image[z_1 * rows * cols + y_i * cols + x_i];
  ip5 = image[z_1 * rows * cols + y_i * cols + x_1];
  ip6 = image[z_1 * rows * cols + y_1 * cols + x_i];
  ip7 = image[z_1 * rows * cols + y_1 * cols + x_1];
  imGc[z_i * rows* cols + y_i * cols + x_i] =
      (ip1 + ip3 + ip5 + ip7 - ip0 - ip2 - ip4 - ip6) / 4;
  imGr[z_i * rows* cols + y_i * cols + x_i] =
      (ip2 + ip3 + ip6 + ip7 - ip0 - ip1 - ip4 - ip5) / 4;
  imGs[z_i * rows* cols + y_i * cols + x_i] =
      (ip4 + ip5 + ip6 + ip7 - ip0 - ip1 - ip2 - ip3) / 4;

}

__kernel void gradients_3d(__global float* image, __global float* imGs, __global float* imGc, __global float* imGr, int slices, int rows, int cols, int frame_index) {
  int z_i = get_global_id(0);
  int y_i = get_global_id(1);
  int x_i = get_global_id(2);

  if (z_i < slices && y_i < rows && x_i < cols) {
    _c_gradient_3d(&image[frame_index*slices*rows*cols], &imGc[0], &imGr[0], &imGs[0], slices, rows, cols, z_i, y_i, x_i);
  }
}

double _c_calculate_dw3D(double distance, double distance_xy, double distance_z,double tSS, double tSS_z) {
  float D_weight_xy, D_weight_z, D_weight;
  D_weight_xy = (exp((-distance_xy * distance_xy) / tSS));
  D_weight_z = (exp((-distance_z * distance_z) / tSS_z));
  D_weight = distance * (D_weight_xy * D_weight_z);
  return pow(D_weight, 4);
}

double _c_calculate_dw_xy(double distance_xy, double tSS) {
  return pow((distance_xy * exp((-distance_xy * distance_xy) / tSS)), 4);
}

double _c_calculate_dw_z(double distance_z, double tSS_z) {
  return pow((distance_z * exp((-distance_z * distance_z) / tSS_z)), 4);
}

double _c_calculate_dk3D(float Gx, float Gy, float Gz, float dx, float dy, float dz, float distance) {
  float Dk = sqrt((Gy * dz - Gz * dy) * (Gy * dz - Gz * dy) + (Gz * dx - Gx * dz) * (Gz * dx - Gx * dz) + (Gx * dy - Gy * dx) * (Gx * dy - Gy * dx)) / sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
  if (isnan(Dk)) {
    Dk = distance;
  }
  Dk = 1 - Dk / distance;
  return Dk;
}

float _c_calculate_rgc3D(int xM, int yM, int sliceM, __global float* imIntGx, __global float* imIntGy, __global float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float voxel_ratio, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) {

    float vx, vy, vz, Gx, Gy, Gz, dx, dy, dz, dz_real, distance, distance_xy, distance_z, distanceWeight, GdotR, Dk;

    float xc = (xM) / magnification_xy;
    float yc = (yM) / magnification_xy;
    float zc = (sliceM) / magnification_z;

    float RGC = 0;
    float distanceWeightSum = 0;

    int _start = -(int)(2 * fwhm);
    int _end = (int)(2 * fwhm + 1);

    int _start_z = -(int)(2 * fwhm_z);
    int _end_z = (int)(2 * fwhm_z + 1);
    
    for (int j = _start; j <= _end; j++) {
        vy = yc + j;

        if (0 < vy && vy < rowsM/magnification_xy - 1) {
            for (int i = _start; i <= _end; i++) {
                vx = xc + i;

                if (0 < vx && vx < colsM/magnification_xy - 1) {
                    for (int k = _start_z; k <= _end_z; k++) {
                        vz = zc + k;

                        if (0 < vz && vz < slicesM/magnification_z - 1) {
                            dx = vx - xc;
                            dy = vy - yc;
                            dz = vz - zc; 
                            dz_real = dz * voxel_ratio;
                            distance = sqrt(dx * dx + dy * dy + dz_real * dz_real);
                            distance_xy = sqrt(dx * dx + dy * dy);
                            distance_z = dz_real;
                            
                            if (distance != 0 && distance_xy <= tSO && distance_z <= tSO_z) {
                                int linear_index = (int)(vz * magnification_z) * rowsM * colsM +
                                                   (int)(magnification_xy * vy) * colsM +
                                                   (int)(magnification_xy * vx);

                                Gx = imIntGx[linear_index];
                                Gy = imIntGy[linear_index];
                                Gz = imIntGz[linear_index];

                                distanceWeight = _c_calculate_dw3D(distance, distance_xy, distance_z, tSS, tSS_z);
                                
                                distanceWeightSum += distanceWeight;
                                GdotR = Gx*dx + Gy*dy + Gz*dz_real;

                                if (GdotR < 0) {
                                    Dk = _c_calculate_dk3D(Gx, Gy, Gz, dx, dy, dz_real, distance);
                                    RGC += (Dk * distanceWeight);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    RGC /= distanceWeightSum;

    if (RGC >= 0 && sensitivity > 1) {
        RGC = pow(RGC, sensitivity);
    } else if (RGC < 0) {
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

  row = row + fwhm*2*magnification_xy;
  col = col + fwhm*2*magnification_xy;
  s = s + fwhm_z*2*magnification_z;

  if (doIntensityWeighting == 1) {
    tmp_slice[s * rowsM * colsM + row * colsM + col] = _c_calculate_rgc3D(col, row, s, &imIntGx[0], &imIntGy[0], &imIntGz[0], colsM, rowsM, slicesM, magnification_xy, magnification_z, ratio_px, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity) * imM[f * nPixels_out + s * rowsM * colsM + row * colsM + col];
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
    output[current_index] = current_slice[current_index];
  } else {
    // Accumulate the average
    output[current_index] += (current_slice[current_index] - output[current_index]) / (frame_i + 1);
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

        // Update the running variance
        variance_output[current_index] += delta * (current_slice[current_index] - mean_output[current_index]);
    }
}