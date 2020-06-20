import numpy as np
import pandas as pd
from nose import with_setup
from lgnpy import LinearGaussian,LinearGaussianExperimental,GaussianBP


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
    lg.set_edge("A", "B")
    lg.set_edge("C", "B")
    lg.set_edge("B", "D")

    assert lg.get_edges() == [("A", "B"), ("B", "D"), ("C", "B")]
    assert lg.get_nodes() == ["A", "B", "C", "D"]
    assert lg.get_children("D") == []
    assert lg.get_children("B") == ["D"]
    assert lg.get_parents("B") == ["A", "C"]

    lg = LinearGaussian()
    lg.set_edges_from([("A", "C"), ("B", "C"), ("C", "E"), ("D", "E")])
    assert lg.get_edges() == [("A", "C"), ("C", "E"), ("B", "C"), ("D", "E")]
    assert lg.get_nodes() == ["A", "C", "B", "E", "D"]
    assert lg.get_parents("C") == ["A", "B"]
    assert lg.get_children("C") == ["E"]



@with_setup(setup,teardown)
def test_parameters():
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


@with_setup(setup, teardown)
def test_LinearGaussian_inference():
    lg1 = LinearGaussian()
    lg1.set_edges_from([('A', 'E'), ('B', 'E'), ('E', 'F'), ('E', 'G'), ('D', 'G'), ('C', 'H'), ('H', 'G')])

    np.random.seed(42)
    n = 100
    data = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E'])

    # Root nodes data
    data['A'] = np.random.normal(5, 2, n)
    data['B'] = np.random.normal(10, 2, n)
    data['C'] = np.random.normal(50, 2, n)
    data['D'] = np.random.normal(20, 2, n)

    # Data for nodes with parents
    data['E'] = 2 * data['A'] + 3 * data['B'] + np.random.normal(3, 5, n)
    data['F'] = 3 * data['E'] + np.random.normal(-5, 2, n)
    data['H'] = 0.4 * data['C'] + np.random.normal(0, 2, n)
    data['G'] = 5 + 3 * data['E'] + 0.5 * data['H'] + 0.2 * data['D'] + np.random.normal(0, 5, n)

    lg1.set_data(data)
    lg1.set_evidences({'A': 5, 'B': 10})
    lg1.run_inference(debug=True)

    summary = lg1.inf_summary

    assert summary.loc['A','Evidence'] == 5.0
    assert summary.loc['G', 'Evidence'] == ''
    np.testing.assert_almost_equal(summary.loc['B','Mean'],10.0446,decimal=3)
    np.testing.assert_almost_equal(summary.loc['H','Mean'],20.1001,decimal=3)

    np.testing.assert_almost_equal(summary.loc['A','Mean_inferred'],5.0, decimal=3)
    np.testing.assert_almost_equal(summary.loc['G', 'Mean_inferred'], 146.9538, decimal=3)

    np.testing.assert_almost_equal(summary.loc['D','Variance'],3.127, decimal=3)
    np.testing.assert_almost_equal(summary.loc['H', 'Variance'], 4.888, decimal=3)

    assert summary.loc['C', 'Variance_inferred'] == ''
    np.testing.assert_almost_equal(summary.loc['G', 'Variance_inferred'], 23.2971, decimal=3)

    np.testing.assert_almost_equal(summary.loc['A', 'u_%change'],4.3339, decimal=3)
    np.testing.assert_almost_equal(summary.loc['E', 'u_%change'],0.4512, decimal=3)
    assert summary.loc['C', 'u_%change'] == ''
    pass

@with_setup(setup, teardown)
def test_LinearGaussianExperimental_inference():
    lg2 = LinearGaussianExperimental()
    lg2.set_edges_from([('A', 'E'), ('B', 'E'), ('E', 'F'), ('E', 'G'), ('D', 'G'), ('C', 'H'), ('H', 'G')])
    np.random.seed(42)
    n = 100
    data = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E'])

    # Root nodes data
    data['A'] = np.random.normal(5, 2, n)
    data['B'] = np.random.normal(10, 2, n)
    data['C'] = np.random.normal(50, 2, n)
    data['D'] = np.random.normal(20, 2, n)

    # Data for nodes with parents
    data['E'] = 2 * data['A'] + 3 * data['B'] + np.random.normal(3, 5, n)
    data['F'] = 3 * data['E'] + np.random.normal(-5, 2, n)
    data['H'] = 0.4 * data['C'] + np.random.normal(0, 2, n)
    data['G'] = 5 + 3 * data['E'] + 0.5 * data['H'] + 0.2 * data['D'] + np.random.normal(0, 5, n)

    lg2.set_data(data)
    lg2.set_evidences({'A': 5, 'B': 10})
    lg2.run_inference('D',debug=True)

    summary = lg2.inf_summary
    print(summary)

    assert summary.loc['A', 'Evidence'] == 5.0
    assert summary.loc['G', 'Evidence'] == ''
    #np.testing.assert_almost_equal(summary.loc['B', 'Mean'], 10.0446, decimal=3)
    np.testing.assert_almost_equal(summary.loc['H', 'Mean'], 20.1001, decimal=3)

    np.testing.assert_almost_equal(summary.loc['A', 'Mean_inferred'], 5.0, decimal=3)
    np.testing.assert_almost_equal(summary.loc['G', 'Mean_inferred'],146.4092, decimal=3)

    np.testing.assert_almost_equal(summary.loc['D', 'Variance'], 3.127, decimal=3)
    np.testing.assert_almost_equal(summary.loc['H', 'Variance'], 4.888, decimal=3)

    assert summary.loc['C', 'Variance_inferred'] == ''
    np.testing.assert_almost_equal(summary.loc['G', 'Variance_inferred'], 23.299, decimal=3)

    np.testing.assert_almost_equal(summary.loc['A', 'u_%change'], 4.3339, decimal=3)
    np.testing.assert_almost_equal(summary.loc['E', 'u_%change'], 0.025, decimal=3)
    assert summary.loc['C', 'u_%change'] == ''
    pass

@with_setup(setup, teardown)
def test_GaussianBP_inference():
    lg3 = GaussianBP()
    lg3.set_edges_from([('A', 'E'), ('B', 'E'), ('E', 'F'), ('E', 'G'), ('D', 'G'), ('C', 'H'), ('H', 'G')])

    np.random.seed(42)
    n = 100
    data = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E'])

    # Root nodes data
    data['A'] = np.random.normal(5, 2, n)
    data['B'] = np.random.normal(10, 2, n)
    data['C'] = np.random.normal(50, 2, n)
    data['D'] = np.random.normal(20, 2, n)

    # Data for nodes with parents
    data['E'] = 2 * data['A'] + 3 * data['B'] + np.random.normal(3, 5, n)
    data['F'] = 3 * data['E'] + np.random.normal(-5, 2, n)
    data['H'] = 0.4 * data['C'] + np.random.normal(0, 2, n)
    data['G'] = 5 + 3 * data['E'] + 0.5 * data['H'] + 0.2 * data['D'] + np.random.normal(0, 5, n)


    lg3.set_data(data)
    lg3.set_evidences({'A': 5, 'B': 10})
    lg3.run_inference(iterations=100)
    #lg3.plot_errors()

    summary = lg3.inf_summary

    assert summary.loc['A', 'Evidence'] == 5.0

    np.testing.assert_almost_equal(summary.loc['B', 'Mean'], 10.0446, decimal=3)
    np.testing.assert_almost_equal(summary.loc['H', 'Mean'], 20.1001, decimal=3)
    pass


@with_setup(setup, teardown)
def test_utility_functions():
    lg4 = LinearGaussian()
    lg4.set_edges_from([('A', 'E'), ('B', 'E'), ('E', 'F'), ('E', 'G'), ('D', 'G'), ('C', 'H'), ('H', 'G')])

    np.random.seed(42)
    n = 100
    data = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E'])
    # Root nodes data
    data['A'] = np.random.normal(5, 2, n)
    data['B'] = np.random.normal(10, 2, n)
    data['C'] = np.random.normal(50, 2, n)
    data['D'] = np.random.normal(20, 2, n)

    # Data for nodes with parents
    data['E'] = 2 * data['A'] + 3 * data['B'] + np.random.normal(3, 5, n)
    data['F'] = 3 * data['E'] + np.random.normal(-5, 2, n)
    data['H'] = 0.4 * data['C'] + np.random.normal(0, 2, n)
    data['G'] = 5 + 3 * data['E'] + 0.5 * data['H'] + 0.2 * data['D'] + np.random.normal(0, 5, n)

    np.random.seed(42)
    n = 100
    data = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E'])

    # Root nodes data
    data['A'] = np.random.normal(5, 2, n)
    data['B'] = np.random.normal(10, 2, n)
    data['C'] = np.random.normal(50, 2, n)
    data['D'] = np.random.normal(20, 2, n)

    # Data for nodes with parents
    data['E'] = 2 * data['A'] + 3 * data['B'] + np.random.normal(3, 5, n)
    data['F'] = 3 * data['E'] + np.random.normal(-5, 2, n)
    data['H'] = 0.4 * data['C'] + np.random.normal(0, 2, n)
    data['G'] = 5 + 3 * data['E'] + 0.5 * data['H'] + 0.2 * data['D'] + np.random.normal(0, 5, n)

    lg4.set_data(data)

    lg4.network_summary()
    #assert lg4.draw_network('Hello',open=False) == None
    #assert lg4.plot_distributions() == None
    plt.close('all')
    pass












