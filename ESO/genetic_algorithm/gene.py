#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random


class Gene:
    """
    """

    def __init__(self, spec_height: int, min_position: int,
                 max_position: int, min_height: int,
                 max_height: int,  band_position: int = None,
                 band_height: int = None) -> None:
        """
        Parameters
        ----------
        spec_height : int
            The height of the spectogram
        min_position : int
            The minimal position of a band in the spectogram
        max_position : int
            The maximal position of a band in the spectogram.
        min_height : int
            The minimal height of a band in the spectogram.
        max_height : int
            The maximal height of a band in the spectogram.
        band_position : int, optional
            The position of the band in the spectogram. 
            The default is None.
        band_height : int, optional
            The height of the band in the spectogram. 
            The default is None.

        Returns
        -------
        None

        """

        # Assumption:
        #    0 <= min_position < max_height < max_position < spec_height

        if max_height == -1:
            raise ValueError("max_height cannot be -1")

        if min_position == -1:
            min_position = 0
        if max_position == -1:
            max_position = spec_height

        if min_height == -1:
            min_height = 1  # 0 is not a valid height

        self.spec_height = spec_height
        self.min_position = min_position
        self.max_position = max_position
        self.min_height = min_height
        self.max_height = max_height

        # Generate a completely random gene
        if (band_position is None) and (band_height is None):
            self._init_random_gene()

        # Generate a partially random gene (fixed band height)
        elif (band_position is None) and (band_height is not None):
            self._init_random_pos_gene(band_height)

        # Generate a partially random gene (fixed band position)
        elif (band_position is not None) and (band_height is None):
            self._init_random_height_gene(band_position)

        # Generate a gene with given band_position and band_height
        else:
            self._init_set_gene(band_position, band_height)

    def _init_random_gene(self) -> None:
        """
        Helper method used to create a completely random Gene.

        Returns
        -------
        None

        """
        # I guess it makes no sense for the height of a band to be 0.
        rand_height = random.randint(self.min_height, self.max_height)

        # for the band position generate an int between (0, self.max_position-rand_height)
        # This ensures that (rand_height + rand_pos) <= max_position
        rand_pos = random.randint(
            self.min_position, self.max_position-rand_height)

        self.band_position = rand_pos
        self.band_height = rand_height

    def _init_random_pos_gene(self, band_height: int) -> None:
        """
        Helper method used to create a partially random Gene of known height.

        Parameters
        ----------
        band_height : int
            The height of the band that should be created.

        Returns
        -------
        None

        """
        assert self.min_height <= band_height, "min_height must be less than or equal to band_height"
        assert band_height <= self.max_height, "band_height must be less than or equal to max_height"

        self.band_height = band_height
        self.band_position = random.randint(
            self.min_position, self.max_position-self.band_height)

    def _init_random_height_gene(self, band_position: int) -> None:
        """
        Helper method used to create a partially random Gene of known position.

        Parameters
        ----------
        band_position : int
            The position of the band that should be created.

        Returns
        -------
        None

        """
        # Assumption band_position is valid: 0 < band_position < max_position
        assert self.min_position <= band_position, "min_position should be less than or equal to band_position"
        assert band_position < self.max_position, "band_position should be less than max_position"

        self.band_position = band_position

        upper_height = min(self.max_position -
                           self.band_position, self.max_height)
        rand_height = random.randint(self.min_height, upper_height)
        self.band_height = rand_height

    def _init_set_gene(self, band_position: int, band_height: int) -> None:
        """
        Helper method used to create a Gene with given parameters.

        Parameters
        ----------
        band_position : int
            The position of the band that should be created.
        band_height : int
            The height of the band that should be created.

        Returns
        -------
        None

        """
        assert self.min_position <= band_position, \
            "min_position should be less than or equal to band_position"
        assert band_position < self.max_position, \
            "band_position should be less than max_position"
        assert self.min_height <= band_height, \
            "min_height should be less than or equal to band_height"
        assert band_height <= self.max_height, \
            "band_height should be less than or equal to max_height"

        # Check boundary condition
        if (band_position + band_height) <= self.max_position:
            self.band_position = band_position
            self.band_height = band_height
        else:
            raise ValueError("The band cannot exceed max_position")

    def __repr__(self) -> str:
        """

        Returns
        -------
        str
            A string representation of a Gene.

        """

        return f"Gene({self.band_position}, {self.band_height})"

    def __str__(self) -> str:
        """

        Returns
        -------
        str
            A string representation of a Gene for printing.

        """

        return f"({self.band_position}, {self.band_height})"

    def get_band_position(self) -> int:
        """

        Returns
        -------
        int
            The position of the band in the spectogram.

        """

        return self.band_position

    def get_band_height(self) -> int:
        """

        Returns
        -------
        int
            The height of the band in the spectogram.

        """

        return self.band_height

    def set_band_position(self, new_band_position: int) -> None:
        """


        Parameters
        ----------
        new_band_position : int
            The new position at which the band should be set.

        Returns
        -------
        None

        """
        assert self.min_position <= new_band_position, \
            "new_band_position should be greater than or equal to min_position"
        assert new_band_position <= (self.max_position - self.band_height), \
            "new_band_position should be less than or equal to (max_position - band_height)"

        self.band_position = new_band_position

    def set_band_height(self, new_band_height: int) -> None:
        """


        Parameters
        ----------
        new_band_height : int
            The new height of the band.

        Returns
        -------
        None

        """
        upper_height = min(self.max_position -
                           self.band_position, self.max_height)

        assert self.min_height <= new_band_height, \
            "new_band_height should be greater than or equal to min_height"
        assert new_band_height <= upper_height, \
            "new_band_height should be less than or equal to min(max_position-band_position, max_height)"

        self.band_height = new_band_height


if __name__ == '__main__':

    print("Testing the Gene Creation Process...")

    print("====================================")

    # Completely random gene (band_position=None, band_height=None)
    gene_1 = Gene(spec_height=200, min_position=0, max_position=128,
                  min_height=1, max_height=10)
    print(f"Gene 1: {gene_1}")

    # Fixed gene (band_position and band_height given)
    gene_2 = Gene(spec_height=200, min_position=0, max_position=128,
                  min_height=1, max_height=10, band_position=80, band_height=7)
    print(f"Gene 2: {gene_2}")

    # Partially random gene (given band_position)
    gene_3 = Gene(spec_height=200, min_position=0,  max_position=128,
                  min_height=1, max_height=10, band_position=100)
    print(f"Gene 3: {gene_3}")

    # Partially random gene (given band_height)
    gene_4 = Gene(spec_height=200, min_position=0, max_position=128,
                  min_height=1, max_height=10, band_height=9)
    print(f"Gene 4: {gene_4}")

    # Should error die silently ?
    # Should we simply fail ?
    # Should we alert the user
    # Right now we are generating a random/partially random gene when some
    # are not valid
    # Could also raise ValueErrors.

    print('\n======================================')
    # Problematic cases
    print("Problematic cases")

    # band_position < 0
    # gene_7 = Gene(spec_height=200, min_position=0, max_position=128,
    #              min_height=1, max_height=10, band_position=-123)
    # print(f"Gene 7: {gene_7}")

    # band_position > max_position
    # gene_8 = Gene(spec_height=200, min_position=0, max_position=128,
    #              min_height=1, max_height=10, band_position=200)

    # print(f"Gene 8: {gene_8}")

    # band_height < 0
    # gene_9 = Gene(spec_height=200, min_position=0, max_position=128,
    #              min_height=1, max_height=10, band_position=90, band_height=-10)
    # print(f"Gene 9: {gene_9}")

    # gene_10 = Gene(spec_height=200, min_position=0, max_position=128,
    #               min_height=1, max_height=10, band_height=-4)
    # print(f"Gene 10: {gene_10}")

    # band_position < 0 and band_height < 0
    # gene_11 = Gene(spec_height=200, min_position=0, max_position=128,
    #               min_height=1, max_height=10, band_position=-120, band_height=-8)
    # print(f"Gene 11: {gene_11}")

    # band_position < 0 and band_height > 0
    # gene_12 = Gene(spec_height=200, min_position=0,
    #               max_position=128, min_height=1,
    #               max_height=10, band_position=-100, band_height=5)

    # band_height > max_height
    # gene_13 = Gene(spec_height=200, min_position=0, max_position=128,
    #               min_height=1, max_height=10, band_height= 11)
