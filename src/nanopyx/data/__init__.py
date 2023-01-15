"""
This collection of modules contain helper functions for downloading and loading example test data.

>>> from nanopyx.data.download import ExampleDataManager
>>> downloader = ExampleDataManager()
>>> datasets = downloader.list_datasets()
>>> datasets
['LongContinuous_UtrGFP', 'PumpyCost7_UtrGFP', 'SMLMS2013_HDTubulinAlexa647', 'SM_U2OS_Pereira_MTAlexaA647_V1', 'ShortContinuous_UtrGFP', 'EMBL_TubulinGFP', 'ImmunoSynapseFormation_LifeActGFP', 'SM_U2OS_Pereira_MTAlexaA647_V2']

"""