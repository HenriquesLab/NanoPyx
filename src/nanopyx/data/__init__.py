"""
Contains helper methods for downloading and loading example test data

>>> from nanopyx.data.download import ExampleDataManager
>>> downloader = ExampleDataManager()
>>> datasets = downloader.list_datasets()
>>> print("\\n".join([" - "+dataset for dataset in datasets]))
 - Cell1_STORM_COS7_IC
 - Cell2_STORM_COS7_IC
 - Cell2_TIRF_COS7_IC
 - ImmunoSynapseFormation_LifeActGFP
 - LongContinuous_UtrGFP
 - MTsGFP_EMBO_AdvMicroscopy_2018
 - PumpyCos7_UtrGFP
 - SMLMS2013_HDTubulinAlexa647
 - SM_U2OS_Pereira_MTAlexaA647_V1
 - SM_U2OS_Pereira_MTAlexaA647_V2
 - ShortContinuous_UtrGFP
 - VirusMapper_ViralParticles_Channel1
 - VirusMapper_ViralParticles_Channel2
 - VirusMapper_ViralParticles_Frontal_Seed-Average_Ch1
 - VirusMapper_ViralParticles_Frontal_Seed-Average_Ch2
 - VirusMapper_ViralParticles_Sagittal_Seed-Average_Ch1
 - VirusMapper_ViralParticles_Sagittal_Seed-Average_Ch2
>>> thumbnail_path = downloader.get_thumbnail("LongContinuous_UtrGFP")
>>> import os
>>> "/".join(["..."]+thumbnail_path.split(os.path.sep)[-5:])
'.../nanopyx/data/examples/LongContinuous_UtrGFP/thumbnail.jpg'
"""
