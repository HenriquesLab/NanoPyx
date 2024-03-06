import numpy as np

from nanopyx.core.generate.simulate_particle_field import (
    get_closest_distance,
    render_particle_histogram,
    simulate_particle_field_based_on_2D_PDF,
)


def test_simulate_particle_field_based_on_2D_PDF(): #(plt):
    """Test the particle field simulation."""

    # Create a 2D PDF
    w = 64
    h = 64
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)
    image_pdf = np.exp(
        -((X - 0.5) ** 2) / 0.1**2 - (Y - 0.5) ** 2 / 0.1**2
    )
    image_pdf = image_pdf.astype(np.float32)

    # Simulate the particle field
    particle_field, mean_closest_distance = simulate_particle_field_based_on_2D_PDF(
        image_pdf, min_particles=1, max_particles=1000
    )

    image_particle_field = render_particle_histogram(particle_field, w, h, 1, 1, 1)

    # # Plot the results
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(image_pdf, interpolation="none")
    # axarr[0].set_title("PDF")
    # axarr[0].set_xlabel("X")
    # axarr[0].set_ylabel("Y")
    # axarr[1].imshow(image_particle_field, interpolation="none")
    # axarr[1].set_title("Particle field")
    # axarr[1].set_xlabel("X")
    # axarr[1].set_ylabel("Y")


def test_simulate_particle_field_ensure_thresholds(): #(plt):
    """Test the particle field simulation."""

    image_pdf = np.ones((100, 100), dtype=np.float32)

    # Simulate the particle field
    (
        particle_field,
        mean_closest_distance,
    ) = simulate_particle_field_based_on_2D_PDF(
        image_pdf,
        min_particles=10,
        max_particles=1000,
        min_distance=0.01,
        mean_distance_threshold=0.1,
        normalize = 1
    )

    assert mean_closest_distance > 0.1
    assert get_closest_distance(particle_field) > 0.01

    image_particle_field = render_particle_histogram(particle_field, 100, 100, 1, 1, 1)

    # # Plot the results
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(image_pdf, interpolation="none")
    # axarr[0].set_title("PDF")
    # axarr[0].set_xlabel("X")
    # axarr[0].set_ylabel("Y")
    # axarr[1].imshow(image_particle_field, interpolation="none")
    # axarr[1].set_title("Particle field")
    # axarr[1].set_xlabel("X")
    # axarr[1].set_ylabel("Y")
