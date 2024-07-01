from nanopyx.core.utils.timeit import timeit, timeit2


def test_timeit():
    @timeit
    def f():
        pass

    @timeit2
    def f2():
        pass

    f()
    f2()
