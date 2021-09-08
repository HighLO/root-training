import ROOT
import numpy as np


def add_array_to_rdf(rdf, name, arr):
    """My helper function to add an array as a column to an RDataFrame.
    The
    """

    if ROOT.IsImplicitMTEnabled():
        raise RuntimeError("add_array_to_rdf doesn't support multithreading with ImplicitMT.")

    if not hasattr(ROOT, "_RDFNumpyAdapterFunctor"):
        ROOT.gInterpreter.Declare(
            """

        template<typename T>
        class _RDFNumpyAdapterFunctor {

        public:
        _RDFNumpyAdapterFunctor(T * arr) : arr_{arr} {}

        T operator()(unsigned int, ULong64_t iEntry) const {
            return arr_[iEntry];
        }
        private:
        T * arr_;

        };
        """
        )

    if not hasattr(ROOT, "_RDFDefineNumpyColumn"):
        ROOT.gInterpreter.Declare(
            """
        template<class DataFrame, typename Functor>
        ROOT::RDF::RNode _RDFDefineNumpyColumn(DataFrame& df, std::string const& name, Functor const& functor) {
           return df.DefineSlotEntry(name, functor, {});
        }
        """
        )

    if not hasattr(rdf, "rdfentry_"):
        rdf.rdfentry_ = rdf.AsNumpy(["rdfentry_"])["rdfentry_"]

    tmp = np.zeros(int(np.max(rdf.rdfentry_[-1])) + 1, dtype=arr.dtype)
    tmp[rdf.rdfentry_] = arr

    type_dict = {"int32": "int"}

    if tmp.dtype.name in type_dict:
        functor = ROOT._RDFNumpyAdapterFunctor[type_dict[tmp.dtype.name]](tmp)
    else:
        raise NotImplemented

    rdf_out = ROOT._RDFDefineNumpyColumn(rdf, ROOT.std.string(name), functor)

    # attempt to keep the input array and the functor alive as long as possible
    keepalive = [tmp, functor]
    for r in [rdf, rdf_out]:
        if not hasattr(r, "keepalive_"):
            r.keepalive_ = []
        r.keepalive_ += keepalive

    return rdf_out
