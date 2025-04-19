from ryd_numerov.elements.element import Element


class Rubidium(Element):
    species = "Rb"
    s = 1 / 2
    ground_state_shell = (5, 0)

    # _ionization_energy = (1_010_029.164_6, 0.000_3, "GHz")  # noqa: ERA001
    # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.83.052515

    _ionization_energy = (4.177_13, 0.000_002, "eV")
    # https://webbook.nist.gov/cgi/inchi?ID=C7440177&Mask=20
    # http://webdelprofesor.ula.ve/ciencias/isolda/libros/handbook.pdf
