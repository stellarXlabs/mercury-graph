import inspect


class BaseClass:
    """
    Base class for common functions and utilities
    """

    def __repr__(self):
        params = ", ".join([f"{k}={v}" for k, v in self.get_params().items()])

        return f"{self.__class__.__name__}({params})"

    def __str__(self):
        params = ", ".join([f"{k}={v}" for k, v in self.get_params().items()])

        base_str = [
            f"Object of class {self.__class__.__name__}.",
            f"",
            f"Initialization parameters: {params}",
        ]

        return "\n".join(base_str)

    def get_params(self):
        """
        Extract a dictionary of the object's parameters

        Returns:
            params (dict): Dictionary of parameters ({name: value})
        """
        # Extract parameter names from the constructor
        init_signature = inspect.signature(self.__init__)
        init_param_names = sorted(
            [
                param_name
                for param_name, _ in init_signature.parameters.items()
                if param_name != "self"
            ]
        )

        # Build dictionary with parameter names and values
        params = {k: getattr(self, k) for k in init_param_names}

        return params
