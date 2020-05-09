import numpy as np
import pandas as pd
from nose import with_setup
from nose.tools import assert_almost_equal


from lgnpy import LinearGaussian

def setup():
    """
    Setup.
    :return: None.
    """
    np.random.seed(42)


def teardown():
    """
    Teardown.
    :return: None.
    """
    pass


@with_setup(setup, teardown)
def test_create_graph():
    """
    Tests creating a basic dag.
    :return: None.
    """
    lg = LinearGaussian()
    lg.set_edge('A','B')
    lg.set_edge('C','B')
    lg.set_edge('B','D')

    assert lg.get_edges() ==  [('A', 'B'), ('B', 'D'), ('C', 'B')]
    assert lg.get_nodes() == ['A', 'B', 'C', 'D']
    assert lg.get_children('D') == []
    assert lg.get_children('B') == ['D']
    assert lg.get_parents('B') == ['A', 'C']

    lg = LinearGaussian()
    lg.set_edges_from([('A', 'C'), ('B', 'C'), ('C', 'E'), ('D', 'E')])
    assert lg.get_edges() == [('A', 'C'), ('C', 'E'), ('B', 'C'), ('D', 'E')]
    assert lg.get_nodes() == ['A', 'C', 'B', 'E', 'D']
    assert lg.get_parents('C') == ['A', 'B']
    assert lg.get_children('C') == ['E']


@with_setup(setup,teardown)
def test_learning_and_inference():
    data = pd.DataFrame(columns=['A','B','C','D','E'])
    n=100
    data['A'] = np.random.normal(0,2,n)
    data['B'] = np.random.normal(5,3,n)
    data['C'] = 2*data['A'] + 3*data['B'] + np.random.normal(0,2,n)
    data['D'] = np.random.normal(2,2,n)
    data['E'] = 3*data['C'] + np.random.normal(0,2,n)

    lg = LinearGaussian()
    lg.set_edges_from([('A', 'C'), ('B', 'C'), ('C', 'E'), ('D', 'E')])
    lg.set_data(data)

    np.testing.assert_almost_equal(list(lg.get_mean()),
                                   [-0.20769303478818774,
                                    5.066913761149772,
                                    14.91514772007384,
                                    2.213680241394609,
                                    44.63343421920083],
                                    decimal=3)

    np.testing.assert_almost_equal(list(np.diag(np.array(lg.get_covariance()))),
                                   [3.29907957452064,
                                    8.18536047354675,
                                    84.70325221116876,
                                    3.126541257985881,
                                    778.505141031734],
                                    decimal = 3)

    np.testing.assert_almost_equal(list(np.diag(np.array(lg.get_precision_matrix()))),
                                   [1.422843910211299,
                                    2.141985350397759,
                                    2.29749879236002,
                                    0.34555297789630945,
                                    0.24461016431093713],
                                   decimal=3)

    lg.set_evidences({'A': 6})
    assert lg.get_evidences() == {'A': 6, 'B': None, 'C': None, 'D': None, 'E': None}

    lg.clear_evidences()
    assert lg.get_evidences() == {'A': None, 'B': None, 'C': None, 'D': None, 'E': None}

    lg.set_evidences({'A': 7, 'B': 2})
    assert lg.get_evidences() == {'A': 7, 'B': 2, 'C': None, 'D': None, 'E': None}

    m,v = lg.run_inference(debug=False)
    np.testing.assert_almost_equal(list(m.values()),
                                   [7, 2, 21.784476453616936, 2.213680241394609, 65.4286632872412],
                                   decimal=3)

    np.testing.assert_almost_equal(v['C'],4.530868459203305,decimal=3)
    np.testing.assert_almost_equal(v['E'], 4.2558278169652795, decimal=3)


    pc =list(lg.get_model_parameters()['C'].values())
    pe = list(lg.get_model_parameters()['E'].values())
    np.testing.assert_almost_equal(pc,
                                   [[0.21821993825026098, 2.2260893644210045, 2.9918154822098213]],
                                   decimal=3)
    np.testing.assert_almost_equal(pe,
                                   [[-1.1154965220785016, 3.0272578114505975, 0.2696565138830367]],
                                   decimal=3)





