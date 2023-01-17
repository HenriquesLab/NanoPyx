"""
Contains helper methods for downloading and loading example test data

>>> from .download import ExampleDataManager
>>> downloader = ExampleDataManager()
>>> datasets = downloader.list_datasets()
>>> print("\\n".join([" - "+dataset for dataset in datasets]))
 - LongContinuous_UtrGFP
 - PumpyCost7_UtrGFP
 - SMLMS2013_HDTubulinAlexa647
 - SM_U2OS_Pereira_MTAlexaA647_V1
 - ShortContinuous_UtrGFP
 - EMBL_TubulinGFP
 - ImmunoSynapseFormation_LifeActGFP
 - SM_U2OS_Pereira_MTAlexaA647_V2
>>> thumbnail_path = downloader.get_thumbnail("LongContinuous_UtrGFP")
>>> import os
>>> "/".join(["..."]+thumbnail_path.split(os.path.sep)[-5:])
'.../nanopyx/data/examples/LongContinuous_UtrGFP/thumbnail.jpg'
    
"""