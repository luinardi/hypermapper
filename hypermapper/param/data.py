from typing import Optional

import torch


class DataArray:
    """
    Storage class for data. Contains four torch Tensors.
         - parameters_array: x values
         - metrics_array: y values
         - timestamp_array: time values
         - feasible_array: feasibility values
         - std_estimate: estimate of the standard deviation of the noise
    """

    def __init__(
            self,
            parameters_array: torch.Tensor,
            metrics_array: torch.Tensor,
            timestamp_array: torch.Tensor,
            feasible_array: torch.Tensor,
            std_estimate: Optional[torch.Tensor] = torch.Tensor(),
    ):

        self.scalarization_array = None
        self.parameters_array = parameters_array
        self.metrics_array = metrics_array
        self.timestamp_array = timestamp_array
        self.feasible_array = feasible_array
        self.std_estimate = std_estimate

        self._update()

    def cat(self, data_array):
        """
        concatenates with another data_array:
        Input:
            data_array: another DataArray object.
        """

        self.parameters_array = torch.cat((self.parameters_array, data_array.parameters_array), 0)
        self.metrics_array = torch.cat((self.metrics_array, data_array.metrics_array), 0)
        self.timestamp_array = torch.cat((self.timestamp_array, data_array.timestamp_array))
        self.feasible_array = torch.cat((self.feasible_array, data_array.feasible_array)).to(dtype=torch.bool)
        self.std_estimate = torch.cat((self.std_estimate, data_array.std_estimate))

        self._update()

    def slice(self, s: torch.Tensor):
        """
        slices the DataArray
        Input:
            - s: the slice vector
        Returns:
            - data array with the sliced values.
        """
        return DataArray(
            self.parameters_array[s, :],
            self.metrics_array[s, :],
            self.timestamp_array[s],
            (self.feasible_array[s] if self.feasible_array.shape[0] > 0 else self.feasible_array),
        )

    def get_feasible(self):
        """
        Returns:
            - A new DataArray with only feasible values.
        """
        if self.feasible_array.shape[0] > 0:
            return DataArray(
                self.parameters_array[self.feasible_array, :],
                self.metrics_array[self.feasible_array, :],
                self.timestamp_array[self.feasible_array],
                self.feasible_array[self.feasible_array],
            )
        else:
            return self

    def _update(self):
        """
        Updates string dict which is used for checking duplicate solutions and length.
        """
        self.string_dict = {"_".join([str(s) for s in row]): idx for idx, row in enumerate(self.parameters_array)}
        self.len = self.parameters_array.shape[0]

    def copy(self):
        """
        Returns:
            - A new DataArray. note that this is not a deepcopy.
        """
        return DataArray(
            self.parameters_array,
            self.metrics_array,
            self.timestamp_array,
            self.feasible_array,
            self.std_estimate,
        )

    def set_scalarization(self, scalarization_weights: torch.Tensor):
        """
        sets the scalarization array
        Input:
            - scalarization_weights: weights for the different metrics in the scalarization
        """
        self.scalarization_array = self.metrics_array @ scalarization_weights

    def __repr__(self):
        return f"\n{self.parameters_array}\n{self.metrics_array}\n{self.timestamp_array}\n{self.feasible_array}\n"

    def __str__(self):
        return self.__repr__()
