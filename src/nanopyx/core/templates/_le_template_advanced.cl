void _c_template(__global float* image);

void _c_template(__global float* image) {
    // add your code logic here
        }
    


__kernel void template(__global float* image)
        {
            int f = get_global_id(0);

            _c_template(&image[f]);
        }