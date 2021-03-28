"""Test to check whether forward moving average is calculated correctly.

"""
import numpy as np
import numpy.testing
import pandas as pd
from utils import create_moving_average


def test_create_moving_average():
    df = generate_input()
    result = create_moving_average(
        data=df, varlist=["var_list"], grouping_var="group", kind="forward", time=7
    )
    test = result["var_list_avg_7d"].head(7)
    np.testing.assert_array_almost_equal(
        [
            -11.42857143,
            -20.42857143,
            -29.0,
            -37.14285714,
            -47.14285714,
            -52.57142857,
            -49.14285714,
        ],
        test.values,
    )


def generate_input():
    data = np.array(
        [
            23.0,
            6.0,
            0.0,
            4.0,
            2.0,
            -60.0,
            -55.0,
            -40,
            -54.0,
            -57.0,
            -66.0,
            -36.0,
            -36.0,
            3.0,
            23.0,
            6.0,
            0.0,
            4.0,
            2.0,
            -60.0,
            -55.0,
            -40,
            -54.0,
            -57.0,
            -66.0,
            -36.0,
            -36.0,
            3.0,
        ]
    )
    # var_list = ["var_list"]
    index = pd.MultiIndex.from_product(
        [
            ["grouping_var_1", "grouping_var_2"],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ]
    )
    index.set_names(["group", "time"], inplace=True)
    df = pd.DataFrame(data=data, index=index, columns=["var_list"])
    return df
