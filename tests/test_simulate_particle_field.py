from nanopyx.core.particles.simulate_particle_field import simulate_particle_field_based_on_2D_PDF, render_particle_histogram

import numpy as np


def test_simulate_particle_field_based_on_2D_PDF(plt):
    """Test the particle field simulation."""

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    image_pdf = np.exp(-(X - 0.5)**2 / 0.1**2 - (Y - 0.5)**2 / 0.1**2)
    image_pdf = image_pdf.astype(np.float32)

    # Simulate the particle field
    particle_field = simulate_particle_field_based_on_2D_PDF(
        image_pdf, min_particles=100, max_particles=10000)

    image_particle_field = render_particle_histogram(particle_field, 100, 100)

    # Plot the results
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image_pdf, interpolation='none')
    axarr[0].set_title('PDF')
    axarr[0].set_xlabel('X')
    axarr[0].set_ylabel('Y')
    axarr[1].imshow(image_particle_field, interpolation='none')
    axarr[1].set_title('Particle field')
    axarr[1].set_xlabel('X')
    axarr[1].set_ylabel('Y')
