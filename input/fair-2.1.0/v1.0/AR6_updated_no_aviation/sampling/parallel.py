def run_stuff(sample):
    import os
    import pickle
    import warnings
    from pathlib import Path

    from pyam import concat
    from dotenv import load_dotenv
    from silicone.database_crunchers import QuantileRollingWindows, RMSClosest

    # pyam and pandas together give a LOT of junk messages
    warnings.simplefilter("ignore")

    # Get environment variables
    load_dotenv()

    DATADIR = Path(os.getenv("DATADIR"))
    DATAIN = DATADIR.joinpath("data_processed/infilling")

    # straight from Jarmo. CO2 AFOLU, CH4, N2O already done
    # we do this with QRW
    major_variables_list = [
        "AR6 climate diagnostics|Emissions|BC",
        #    "Emissions|CH4",
        #    "Emissions|CO2|AFOLU",
        "AR6 climate diagnostics|Emissions|CO",
        #    "Emissions|N2O",
        "AR6 climate diagnostics|Emissions|NH3",
        "AR6 climate diagnostics|Emissions|NOx",
        "AR6 climate diagnostics|Emissions|OC",
        "AR6 climate diagnostics|Emissions|Sulfur",
        "AR6 climate diagnostics|Emissions|VOC",
    ]

    # We do this with RMS Closest
    main_hfc_pfc_variables_list = [
        "AR6 climate diagnostics|Emissions|HFC|HFC134a",
        "AR6 climate diagnostics|Emissions|HFC|HFC143a",
        "AR6 climate diagnostics|Emissions|HFC|HFC227ea",
        "AR6 climate diagnostics|Emissions|HFC|HFC23",
        "AR6 climate diagnostics|Emissions|HFC|HFC32",
        "AR6 climate diagnostics|Emissions|HFC|HFC43-10",
        #    "AR6 climate diagnostics|Emissions|HFC|HFC245ca",
        "AR6 climate diagnostics|Emissions|HFC|HFC125",
        "AR6 climate diagnostics|Emissions|SF6",
        "AR6 climate diagnostics|Emissions|PFC|CF4",
        "AR6 climate diagnostics|Emissions|PFC|C2F6",
        "AR6 climate diagnostics|Emissions|PFC|C6F14",
    ]

    # We do this with RMS Closest, using SSP emissions database.
    minor_ghg_variables_list = [
        "AR6 climate diagnostics|Emissions|CCl4",
        "AR6 climate diagnostics|Emissions|CFC11",
        "AR6 climate diagnostics|Emissions|CFC113",
        "AR6 climate diagnostics|Emissions|CFC114",
        "AR6 climate diagnostics|Emissions|CFC115",
        "AR6 climate diagnostics|Emissions|CFC12",
        "AR6 climate diagnostics|Emissions|CH2Cl2",
        "AR6 climate diagnostics|Emissions|CH3Br",
        "AR6 climate diagnostics|Emissions|CH3CCl3",
        "AR6 climate diagnostics|Emissions|CH3Cl",
        "AR6 climate diagnostics|Emissions|CHCl3",
        "AR6 climate diagnostics|Emissions|HCFC141b",
        "AR6 climate diagnostics|Emissions|HCFC142b",
        "AR6 climate diagnostics|Emissions|HCFC22",
        "AR6 climate diagnostics|Emissions|HFC|HFC152a",
        "AR6 climate diagnostics|Emissions|HFC|HFC236fa",
        "AR6 climate diagnostics|Emissions|HFC|HFC245fa",  # incorrectly named?
        "AR6 climate diagnostics|Emissions|HFC|HFC365mfc",
        "AR6 climate diagnostics|Emissions|Halon1202",
        "AR6 climate diagnostics|Emissions|Halon1211",
        "AR6 climate diagnostics|Emissions|Halon1301",
        "AR6 climate diagnostics|Emissions|Halon2402",
        "AR6 climate diagnostics|Emissions|NF3",
        "AR6 climate diagnostics|Emissions|PFC|C3F8",
        "AR6 climate diagnostics|Emissions|PFC|C4F10",
        "AR6 climate diagnostics|Emissions|PFC|C5F12",
        "AR6 climate diagnostics|Emissions|PFC|C7F16",
        "AR6 climate diagnostics|Emissions|PFC|C8F18",
        "AR6 climate diagnostics|Emissions|PFC|cC4F8",
        "AR6 climate diagnostics|Emissions|SO2F2",
    ]

    # Get data pickles
    with open(DATAIN.joinpath("infiller_database_ar6.pkl"), "rb") as handle:
        infiller_database = pickle.load(handle)

    with open(DATAIN.joinpath("infiller_database_ssp.pkl"), "rb") as handle:
        infiller_database_ssp = pickle.load(handle)

    with open(DATAIN.joinpath("rff_co2_eip.pkl"), "rb") as handle:
        co2_eip = pickle.load(handle)

    filler_input = co2_eip.filter(scenario="{:05d}".format(sample))
    inner_list = []
    lead = ["AR6 climate diagnostics|Emissions|CO2|Energy and Industrial Processes"]

    # major
    cruncher = QuantileRollingWindows(infiller_database)
    for follow in major_variables_list:
        filler = cruncher.derive_relationship(follow, lead)
        filler_data = filler_input.filter(
            year=infiller_database["year"].unique(), variable=lead
        )
        qrw_infilled = filler(filler_data)
        inner_list.append(qrw_infilled.filter(variable=follow))

    # minor
    cruncher = RMSClosest(infiller_database)
    for follow in main_hfc_pfc_variables_list:
        filler = cruncher.derive_relationship(follow, lead)
        filler_data = filler_input.filter(
            year=infiller_database["year"].unique(), variable=lead
        )
        qrw_infilled = filler(filler_data)
        inner_list.append(qrw_infilled.filter(variable=follow))

    # micro
    cruncher = RMSClosest(infiller_database_ssp)
    for follow in minor_ghg_variables_list:
        filler = cruncher.derive_relationship(follow, lead)
        filler_data = filler_input.filter(
            year=infiller_database["year"].unique(), variable=lead
        )
        qrw_infilled = filler(filler_data)
        inner_list.append(qrw_infilled.filter(variable=follow))

    pyam_output = concat(inner_list)
    return pyam_output
