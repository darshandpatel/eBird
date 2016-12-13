from pyspark import SparkConf, SparkContext
import sys
from data_exploration import DataExploration
from pyspark.mllib.classification import LogisticRegressionModel
import argparse


class DataPrediction:

    header_dict = {}
    drop_list = ["SAMPLING_EVENT_ID", "LOC_ID", "DAY", "COUNTRY", "STATE_PROVINCE", "COUNTY", "COUNT_TYPE", "OBSERVER_ID",
                 "ELEV_GT","ELEV_NED","GROUP_ID","BAILEY_ECOREGION", "OMERNIK_L3_ECOREGION","SUBNATIONAL2_CODE", "LATITUDE", "LONGITUDE"]
    drop_multiples_list = ["NLCD", "CAUS_PREC0", "CAUS_PREC1", "CAUS_SNOW0", "CAUS_SNOW1", "CAUS_TEMP_AVG0", "CAUS_TEMP_AVG1"
                           , "CAUS_TEMP_MIN0", "CAUS_TEMP_MIN1", "CAUS_TEMP_MAX0", "CAUS_TEMP_MAX1"]
    protocol_list = ["P20", "P21", "P22", "P23", "P34", "P35", "P39", "P40", "P41", "P44", "P45", "P46", "P47", "P48",
                     "P49", "P50", "P51", "P52", "P55", "P56"]
    birds_column_ids = None
    drop_column_ids = []
    target_ID = 0
    mean = []
    variance = []

    def __init__(self):
        self.conf = SparkConf()
        self.sc = SparkContext(conf=self.conf)

    def read_test_file(self, file_path):
        return self.sc.textFile(file_path)

    # Validation (Scikit Learn)
    def val_prediction_through_models(self, val_data_set, model_path, prediction_path,
                                      header_dict, birds_columns, drop_index):

        models = self.sc.pickleFile(model_path)
        model_list = models.map(lambda x: x[1]).collect()
        print "model_list ", model_list
        print "model_list type : ", type(model_list)

        model_broadcast = self.sc.broadcast(model_list)

        processed_val_data = (val_data_set.map(lambda x: DataExploration.swap_target(x, header_dict["Agelaius_phoeniceus"])).
                              map(lambda x: (x[header_dict("SAMPLING_EVENT_ID")[0]],
                                             DataExploration.custom_function(x, False, True, header_dict, birds_columns, drop_index))))

        predictions = processed_val_data.map(lambda x: DataPrediction.val_prediction_values(x, model_broadcast))
        predictions.saveAsTextFile(prediction_path)

        return processed_val_data

    # Validation (Scikit Learn)
    def val_lr_prediction_through_models(self, val_data_set, model_path, prediction_path,
                                         header_dict, birds_columns, drop_index):

         model = LogisticRegressionModel.load(self.sc, model_path)
         # model = RandomForestModel.load(self.sc, model_path)
         model_broadcast = self.sc.broadcast(model)

         processed_test_data = val_data_set.map(lambda x: DataExploration.swap_target(x, header_dict["Agelaius_phoeniceus"])).\
          map(lambda x: DataExploration.custom_function(x, False, True, header_dict, birds_columns, drop_index))

         predictions = processed_test_data.map(lambda x: (x[0],
                                                          str(x[1][0]) + ',' +
                                                          str(DataPrediction.val_lr_prediction_values(x[1][1],
                                                                                                    model_broadcast))))
         predictions.saveAsTextFile(prediction_path)

    @staticmethod
    def val_prediction_values(x, model_broadcast):

        prob_sum = 0
        for model in model_broadcast.value:
            #prob_sum += model.predict(x[1].features.toArray().reshape(1, -1))
            prob_sum += model.predict(x[1])

        prediction = 0
        if(prob_sum/ len(model_broadcast.value)) > 0.5:
            prediction = 1

        return "" + x[0] + "," + str(x[1].label) + "," + str(prediction)

    @staticmethod
    def val_lr_prediction_values(x, model):
        prediction = 0
        if model.value.predict(x) > 0.5:
            prediction = 1
        return "" + x[0] + "," + str(x[1].label) + "," + str(prediction)

    # Prediction on Test Data set (Scikit Learn)
    def test_prediction_through_models(self, test_data_set, model_path, prediction_path,
                                       header_dict, birds_columns, drop_index):

        models = self.sc.pickleFile(model_path)
        model_list = models.map(lambda x: x[1]).collect()
        print "model_list ", model_list
        print "model_list type : ", type(model_list)

        model_broadcast = self.sc.broadcast(model_list)

        processed_test_data = test_data_set.zipWithIndex().map(lambda x: (x[1], x[0])).\
            map(lambda x: (x[0], DataExploration.swap_target(x[1], header_dict["Agelaius_phoeniceus"]))).\
            map(lambda x: (x[0], (x[1][DataExploration.get_col_id("SAMPLING_EVENT_ID")[0]],
                                  DataExploration.custom_function(x[1], False, True, header_dict, birds_columns, drop_index))))
        #print processed_test_data.first()
        predictions = processed_test_data.map(lambda x: (x[0],
                                                         str(x[1][0]) + ',' +
                                                         str(DataPrediction.test_prediction_values(x[1][1], model_broadcast))))

        #print predictions.collect()
        label = self.sc.parallelize(['SAMPLING_EVENT_ID,SAW_AGELAIUS_PHOENICEUS'])
        final_results = label.union(predictions.sortBy(lambda x: x[0]).map(lambda x: x[1]).coalesce(1))
        final_results.saveAsTextFile(prediction_path)
        #return predictions

    # Prediction on Test Data set (MLlib Logistic Regression)
    def test_lr_prediction_through_models(self, test_data_set, model_path, prediction_path,
                                          header_dict, birds_columns, drop_index):

        model = LogisticRegressionModel.load(self.sc, model_path)
        #model = RandomForestModel.load(self.sc, model_path)
        model_broadcast = self.sc.broadcast(model)

        processed_test_data = test_data_set.zipWithIndex().map(lambda x: (x[1], x[0])). \
         map(lambda x: (x[0], DataExploration.swap_target(x[1], header_dict["Agelaius_phoeniceus"]))).\
            filter(lambda x : DataExploration.filter_header(x[1], header_dict)).\
            map(lambda x: (x[0], (x[1][header_dict["SAMPLING_EVENT_ID"][0]],
                               DataExploration.custom_function(x[1], False, True, header_dict, birds_columns,
                                                               drop_index))))

        predictions = processed_test_data.map(lambda x: (x[0],
                                                         str(x[1][0]) + ',' +
                                                         str(DataPrediction.test_prediction_values(x[1][1], model_broadcast))))

        #print predictions.collect()
        label = self.sc.parallelize(['SAMPLING_EVENT_ID,SAW_AGELAIUS_PHOENICEUS'])
        final_results = label.union(predictions.sortBy(lambda x: x[0]).map(lambda x: x[1])).coalesce(1)
        final_results.saveAsTextFile(prediction_path)
        #return predictions

    @staticmethod
    def test_prediction_values(x, model_broadcast):
         prob_sum = 0
         for model in model_broadcast.value:
             #prob_sum += model.predict(x.toArray().reshape(1, -1))
             prob_sum += model.predict(x[1])

         prediction = 0
         if (prob_sum / len(model_broadcast.value)) > 0.5:
             prediction = 1
         return prediction

    @staticmethod
    def test_prediction_values(x, model):
         if model.value.predict(x) > 0.5:
             return 1
         else:
             return 0


if __name__ == "__main__":

    data_prediction = DataPrediction()
    parser = argparse.ArgumentParser(description='Process validation and test inputs')

    parser.add_argument('testFile', default="")
    parser.add_argument('modelFile')
    parser.add_argument('resultFile')

    args = parser.parse_args()
    print args
    input_file = None


    input_file = args.testFile

    model_path = args.modelFile
    prediction_path = args.resultFile

    data_set = data_prediction.read_test_file(input_file).persist()

    header_dict = DataExploration.create_header_dict(
        [u'SAMPLING_EVENT_ID', u'LOC_ID', u'LATITUDE', u'LONGITUDE', u'YEAR', u'MONTH', u'DAY', u'TIME', u'COUNTRY',
      u'STATE_PROVINCE', u'COUNTY', u'COUNT_TYPE', u'EFFORT_HRS', u'EFFORT_DISTANCE_KM', u'EFFORT_AREA_HA',
      u'OBSERVER_ID', u'NUMBER_OBSERVERS', u'GROUP_ID', u'PRIMARY_CHECKLIST_FLAG', u'Zenaida_macroura',
      u'Corvus_brachyrhynchos', u'Turdus_migratorius', u'Cardinalis_cardinalis', u'Cyanocitta_cristata',
      u'Sturnus_vulgaris', u'Melospiza_melodia', u'Agelaius_phoeniceus', u'Spinus_tristis', u'Anas_platyrhynchos',
      u'Branta_canadensis', u'Picoides_pubescens', u'Haemorhous_mexicanus', u'Cathartes_aura',
      u'Melanerpes_carolinus', u'Passer_domesticus', u'Ardea_herodias', u'Baeolophus_bicolor', u'Colaptes_auratus',
      u'Poecile_atricapillus', u'Mimus_polyglottos', u'Buteo_jamaicensis', u'Sitta_carolinensis', u'Junco_hyemalis',
      u'Thryothorus_ludovicianus', u'Setophaga_coronata', u'Quiscalus_quiscula', u'Larus_delawarensis',
      u'Phalacrocorax_auritus', u'Charadrius_vociferus', u'Poecile_carolinensis', u'Hirundo_rustica',
      u'Columba_livia', u'Dumetella_carolinensis', u'Molothrus_ater', u'Zonotrichia_albicollis',
      u'Geothlypis_trichas', u'Ardea_alba', u'Spizella_passerina', u'Tachycineta_bicolor', u'Sialia_sialis',
      u'Fulica_americana', u'Sayornis_phoebe', u'Megaceryle_alcyon', u'Larus_argentatus', u'Bombycilla_cedrorum',
      u'Regulus_calendula', u'Pipilo_erythrophthalmus', u'Podilymbus_podiceps', u'Zonotrichia_leucophrys',
      u'Picoides_villosus', u'Troglodytes_aedon', u'Corvus_corax', u'Falco_sparverius', u'Polioptila_caerulea',
      u'Pandion_haliaetus', u'Setophaga_petechia', u'Haliaeetus_leucocephalus', u'Aix_sponsa', u'Vireo_olivaceus',
      u'Bucephala_albeola', u'Buteo_lineatus', u'Egretta_thula', u'Dryocopus_pileatus', u'Passerina_cyanea',
      u'Anas_clypeata', u'Chaetura_pelagica', u'Anas_strepera', u'Passerculus_sandwichensis', u'Circus_cyaneus',
      u'Accipiter_cooperii', u'Coragyps_atratus', u'Pipilo_maculatus', u'Calypte_anna', u'Sayornis_nigricans',
      u'Archilochus_colubris', u'Toxostoma_rufum', u'Tyrannus_tyrannus', u'Streptopelia_decaocto',
      u'Aphelocoma_californica', u'Oxyura_jamaicensis', u'Anas_crecca', u'Sitta_canadensis', u'Spinus_psaltria',
      u'Spizella_pusilla', u'Icterus_galbula', u'Leucophaeus_atricilla', u'Myiarchus_crinitus',
      u'Stelgidopteryx_serripennis', u'Aythya_collaris', u'Anas_americana', u'Contopus_virens', u'Corvus_ossifragus',
      u'Larus_marinus', u'Melospiza_georgiana', u'Catharus_guttatus', u'Tringa_melanoleuca', u'Butorides_virescens',
      u'Actitis_macularius', u'Setophaga_ruticilla', u'Regulus_satrapa', u'Calidris_minutilla',
      u'Thryomanes_bewickii', u'Lophodytes_cucullatus', u'Anas_discors', u'Vireo_gilvus', u'Anas_rubripes',
      u'Oreothlypis_celata', u'Mergus_merganser', u'Mniotilta_varia', u'Sturnella_magna', u'Setophaga_pinus',
      u'Aythya_affinis', u'Vireo_griseus', u'Setophaga_palmarum', u'Quiscalus_mexicanus', u'Spinus_pinus',
      u'Pelecanus_occidentalis', u'Gavia_immer', u'Certhia_americana', u'Euphagus_cyanocephalus',
      u'Setophaga_americana', u'Psaltriparus_minimus', u'Sturnella_neglecta', u'Petrochelidon_pyrrhonota',
      u'Bucephala_clangula', u'Seiurus_aurocapilla', u'Sphyrapicus_varius', u'Mergus_serrator',
      u'Cyanocitta_stelleri', u'Pheucticus_ludovicianus', u'Sterna_forsteri', u'Anas_acuta', u'Tringa_semipalmata',
      u'Zenaida_asiatica', u'Haemorhous_purpureus', u'Meleagris_gallopavo', u'Hylocichla_mustelina',
      u'Cistothorus_palustris', u'Spizella_arborea', u'Nycticorax_nycticorax', u'Melozone_crissalis',
      u'Progne_subis', u'Accipiter_striatus', u'Eremophila_alpestris', u'Lanius_ludovicianus', u'Egretta_caerulea',
      u'Eudocimus_albus', u'Pelecanus_erythrorhynchos', u'Tringa_flavipes', u'Piranga_olivacea',
      u'Larus_californicus', u'Cardellina_pusilla', u'Setophaga_virens', u'Cygnus_olor', u'Passerella_iliaca',
      u'Melospiza_lincolnii', u'Larus_occidentalis', u'Grus_canadensis', u'Pluvialis_squatarola',
      u'Podiceps_auritus', u'Hydroprogne_caspia', u'Aythya_americana', u'Poecile_rufescens', u'Setophaga_magnolia',
      u'Sayornis_saya', u'Charadrius_semipalmatus', u'Tyrannus_verticalis', u'Catharus_ustulatus',
      u'Quiscalus_major', u'Himantopus_mexicanus', u'Pica_hudsonia', u'Calidris_alba', u'Melanerpes_formicivorus',
      u'Tachycineta_thalassina', u'Egretta_tricolor', u'Oreothlypis_ruficapilla', u'Zonotrichia_atricapilla',
      u'Melanerpes_erythrocephalus', u'Setophaga_pensylvanica', u'Gallinula_galeata', u'Pheucticus_melanocephalus',
      u'Gallinago_delicata', u'Icterus_spurius', u'Sialia_mexicana', u'Coccyzus_americanus', u'Vireo_solitarius',
      u'Anthus_rubescens', u'Aechmophorus_occidentalis', u'Callipepla_californica', u'Bubo_virginianus',
      u'Chroicocephalus_philadelphia', u'Bubulcus_ibis', u'Calidris_alpina', u'Piranga_rubra', u'Falco_peregrinus',
      u'Passerina_caerulea', u'Picoides_nuttallii', u'Recurvirostra_americana', u'Thalasseus_maximus',
      u'Podiceps_nigricollis', u'Anhinga_anhinga', u'Piranga_ludoviciana', u'Poecile_gambeli',
      u'Baeolophus_inornatus', u'Aythya_valisineria', u'Contopus_sordidulus', u'Chondestes_grammacus',
      u'Melanitta_perspicillata', u'Anas_cyanoptera', u'Calidris_pusilla', u'Catharus_fuscescens',
      u'Vireo_flavifrons', u'Arenaria_interpres', u'Aythya_marila', u'Picoides_scalaris', u'Calidris_mauri',
      u'Sitta_pusilla', u'Falco_columbarius', u'Archilochus_alexandri', u'Tringa_solitaria', u'Troglodytes_hiemalis',
      u'Chen_caerulescens', u'Setophaga_caerulescens', u'Empidonax_traillii', u'Icteria_virens',
      u'Phasianus_colchicus', u'Strix_varia', u'Oreothlypis_peregrina', u'Setophaga_striata', u'Sterna_hirundo',
      u'Limnodromus_scolopaceus', u'Riparia_riparia', u'Parkesia_noveboracensis', u'Pooecetes_gramineus',
      u'Icterus_bullockii', u'Empidonax_minimus', u'Setophaga_discolor', u'Limnodromus_griseus',
      u'Setophaga_townsendi', u'Larus_glaucescens', u'Chamaea_fasciata', u'Setophaga_citrina',
      u'Dolichonyx_oryzivorus', u'Porzana_carolina', u'Auriparus_flaviceps', u'Setophaga_dominica',
      u'Empidonax_virescens', u'Tyrannus_forficatus', u'Limosa_fedoa', u'Branta_bernicla', u'Buteo_platypterus',
      u'Patagioenas_fasciata', u'Plegadis_chihi', u'Calidris_melanotos', u'Sternula_antillarum', u'Setophaga_fusca',
      u'Chordeiles_minor', u'Myiarchus_cinerascens', u'Xanthocephalus_xanthocephalus', u'Plegadis_falcinellus',
      u'Anas_fulvigula', u'Gavia_stellata', u'Selasphorus_rufus', u'Sitta_pygmaea', u'Melanitta_fusca',
      u'Columbina_passerina', u'Haematopus_palliatus', u'Branta_hutchinsii', u'Empidonax_difficilis',
      u'Spiza_americana', u'Rynchops_niger', u'Baeolophus_atricristatus', u'Vireo_huttoni', u'Somateria_mollissima',
      u'Elanus_leucurus', u'Clangula_hyemalis', u'Passerina_ciris', u'Vermivora_cyanoptera', u'Nyctanassa_violacea',
      u'Columbina_inca', u'Melanerpes_uropygialis', u'Mycteria_americana', u'Rallus_limicola',
      u'Colinus_virginianus', u'Dendrocygna_autumnalis', u'Passerina_amoena', u'Buteo_swainsoni',
      u'Phalacrocorax_brasilianus', u'Melanerpes_aurifrons', u'Parkesia_motacilla', u'Protonotaria_citrea',
      u'Numenius_americanus', u'Cygnus_columbianus', u'Numenius_phaeopus', u'Platalea_ajaja', u'Selasphorus_sasin',
      u'Troglodytes_pacificus', u'Toxostoma_curvirostre', u'Callipepla_gambelii', u'Tyrannus_vociferans',
      u'Euphagus_carolinus', u'Ammodramus_savannarum', u'Larus_heermanni', u'Anser_albifrons',
      u'Phalacrocorax_penicillatus', u'Phainopepla_nitens', u'Salpinctes_obsoletus', u'Morus_bassanus',
      u'Setophaga_nigrescens', u'Phalaropus_tricolor', u'Buteo_lagopus', u'Melanitta_americana',
      u'Geococcyx_californianus', u'Selasphorus_platycercus', u'Aeronautes_saxatalis', u'Phalacrocorax_pelagicus',
      u'Pyrocephalus_rubinus', u'Campylorhynchus_brunneicapillus', u'Cardellina_canadensis', u'Caracara_cheriway',
      u'Sialia_currucoides', u'Melozone_aberti', u'Myadestes_townsendi', u'Ixoreus_naevius', u'Podiceps_grisegena',
      u'Rallus_crepitans', u'Chlidonias_niger', u'Contopus_cooperi', u'Aechmophorus_clarkii', u'Spizella_pallida',
      u'Megascops_asio', u'Botaurus_lentiginosus', u'Catherpes_mexicanus', u'Toxostoma_redivivum',
      u'Loxia_curvirostra', u'Leucophaeus_pipixcan', u'Aquila_chrysaetos', u'Coccothraustes_vespertinus',
      u'Larus_canus', u'Amphispiza_bilineata', u'Setophaga_tigrina', u'Geothlypis_tolmiei', u'Cygnus_buccinator',
      u'Larus_fuscus', u'Pipilo_chlorurus', u'Helmitheros_vermivorum', u'Icterus_cucullatus', u'Melozone_fusca',
      u'Egretta_rufescens', u'Phalaropus_lobatus', u'Sphyrapicus_ruber', u'Acanthis_flammea',
      u'Ictinia_mississippiensis', u'Vireo_bellii', u'Haemorhous_cassinii', u'Spizella_breweri',
      u'Calidris_himantopus', u'Setophaga_castanea', u'Uria_aalge', u'Bonasa_umbellus', u'Scolopax_minor',
      u'Geothlypis_formosa', u'Zonotrichia_querula', u'Thalasseus_sandvicensis', u'Charadrius_melodus',
      u'Pitangus_sulphuratus', u'Cistothorus_platensis', u'Plectrophenax_nivalis', u'Cardinalis_sinuatus',
      u'Arenaria_melanocephala', u'Aimophila_ruficeps', u'Haematopus_bachmani', u'Calidris_bairdii',
      u'Empidonax_alnorum', u'Cepphus_columba', u'Gavia_pacifica', u'Puffinus_griseus', u'Chaetura_vauxi',
      u'Thalasseus_elegans', u'Tyto_alba', u'Vireo_cassinii', u'Phalacrocorax_carbo', u'Chen_rossii',
      u'Nucifraga_columbiana', u'Calidris_canutus', u'Sphyrapicus_nuchalis', u'Cyanocorax_yncas',
      u'Lanius_excubitor', u'Falco_mexicanus', u'Empidonax_oberholseri', u'Toxostoma_longirostre',
      u'Charadrius_nivosus', u'Vireo_plumbeus', u'Vireo_philadelphicus', u'Calidris_fuscicollis',
      u'Catharus_minimus', u'Geothlypis_philadelphia', u'Calypte_costae', u'Polioptila_melanura',
      u'Ixobrychus_exilis', u'Setophaga_cerulea', u'Cerorhinca_monocerata', u'Baeolophus_wollweberi',
      u'Buteo_regalis', u'Coccyzus_erythropthalmus', u'Athene_cunicularia', u'Empidonax_hammondii',
      u'Bubo_scandiacus', u'Bucephala_islandica', u'Cynanthus_latirostris', u'Aphelocoma_wollweberi',
      u'Aramus_guarauna', u'Tyrannus_couchii', u'Myiopsitta_monachus', u'Oreothlypis_luciae',
      u'Calcarius_lapponicus', u'Petrochelidon_fulva', u'Setophaga_occidentalis', u'Pluvialis_dominica',
      u'Pica_nuttalli', u'Histrionicus_histrionicus', u'Ammodramus_maritimus', u'Parabuteo_unicinctus',
      u'Molothrus_aeneus', u'Gelochelidon_nilotica', u'Vermivora_chrysoptera', u'Anas_penelope',
      u'Puffinus_creatopus', u'Bartramia_longicauda', u'Myiarchus_tyrannulus', u'Larus_thayeri',
      u'Charadrius_wilsonia', u'Ortalis_vetula', u'Leptotila_verreauxi', u'Empidonax_wrightii',
      u'Calamospiza_melanocorys', u'Arremonops_rufivirgatus', u'Cinclus_mexicanus', u'Oreoscoptes_montanus',
      u'Porphyrio_martinicus', u'Corvus_cryptoleucus', u'Fregata_magnificens', u'Asio_flammeus',
      u'Larus_hyperboreus', u'Empidonax_flaviventris', u'Elanoides_forficatus', u'Amazilia_yucatanensis',
      u'Perisoreus_canadensis', u'Melanerpes_lewis', u'Empidonax_occidentalis', u'Stercorarius_parasiticus',
      u'Antrostomus_carolinensis', u'Selasphorus_calliope', u'Calidris_maritima', u'Megascops_kennicottii',
      u'Cepphus_grylle', u'Stercorarius_pomarinus', u'Tachybaptus_dominicus', u'Myioborus_pictus',
      u'Antrostomus_vociferus', u'Larus_glaucoides', u'Ptychoramphus_aleuticus', u'Spinus_lawrencei',
      u'Ammodramus_caudacutus', u'Ammodramus_henslowii', u'Oceanites_oceanicus', u'Fulmarus_glacialis',
      u'Icterus_parisorum', u'Alca_torda', u'Aegolius_acadicus', u'Calidris_virgata', u'Picoides_albolarvatus',
      u'Buteo_plagiatus', u'Agelaius_tricolor', u'Oreothlypis_virginiae', u'Loxia_leucoptera', u'Cairina_moschata',
      u'Baeolophus_ridgwayi', u'Tyrannus_melancholicus', u'Phalaropus_fulicarius', u'Toxostoma_crissale',
      u'Chordeiles_acutipennis', u'Peucaea_cassinii', u'Oreortyx_pictus', u'Pinicola_enucleator',
      u'Callipepla_squamata', u'Icterus_gularis', u'Phoebastria_nigripes', u'Gymnorhinus_cyanocephalus',
      u'Phalaenoptilus_nuttallii', u'Geranoaetus_albicaudatus', u'Lonchura_punctulata', u'Piranga_flava',
      u'Sphyrapicus_thyroideus', u'Rallus_elegans', u'Junco_phaeonotus', u'Accipiter_gentilis',
      u'Chloroceryle_americana', u'Ammodramus_nelsoni', u'Dendrocygna_bicolor', u'Puffinus_gravis', u'Xema_sabini',
      u'Tringa_incana', u'Brachyramphus_marmoratus', u'Ammodramus_leconteii', u'Myiarchus_tuberculifer',
      u'Eugenes_fulgens', u'Rissa_tridactyla', u'Puffinus_opisthomelas', u'Calidris_subruficollis',
      u'Sterna_dougallii', u'Glaucidium_gnoma', u'Calonectris_diomedea', u'Amazona_viridigenalis',
      u'Peucaea_carpalis', u'Setophaga_graciae', u'Asio_otus', u'Sterna_paradisaea', u'Limnothlypis_swainsonii',
      u'Picoides_arizonae', u'Artemisiospiza_belli', u'Limosa_haemastica', u'Colaptes_chrysoides',
      u'Megaceryle_torquata', u'Spizella_atrogularis', u'Oceanodroma_leucorhoa', u'Turdus_grayi',
      u'Bombycilla_garrulus', u'Tyrannus_dominicensis', u'Buteo_albonotatus', u'Peucaea_aestivalis',
      u'Nyctidromus_albicollis', u'Fratercula_arctica', u'Alectoris_chukar', u'Camptostoma_imberbe',
      u'Corvus_caurinus', u'Artemisiospiza_nevadensis', u'Calcarius_ornatus', u'Rallus_obsoletus',
      u'Picoides_borealis', u'Passer_montanus', u'Polioptila_californica', u'Puffinus_bulleri',
      u'Aphelocoma_coerulescens', u'Oceanodroma_melania', u'Aratinga_nenday', u'Brotogeris_chiriri',
      u'Oporornis_agilis', u'Puffinus_puffinus', u'Alopochen_aegyptiaca', u'Cardellina_rubrifrons',
      u'Passerina_versicolor', u'Grus_americana', u'Anthus_spragueii', u'Crotophaga_sulcirostris',
      u'Stercorarius_longicaudus', u'Setophaga_chrysoparia', u'Picoides_arcticus', u'Leucosticte_tephrocotis',
      u'Poecile_hudsonicus', u'Stercorarius_maccormicki', u'Puffinus_lherminieri', u'Buteo_brachyurus',
      u'Fratercula_cirrhata', u'Lampornis_clemenciae', u'Dendragapus_fuliginosus', u'Sula_leucogaster',
      u'Rostrhamus_sociabilis', u'Peucaea_botterii', u'Psittacara_erythrogenys', u'Patagioenas_leucocephala',
      u'Hydrocoloeus_minutus', u'Picoides_dorsalis', u'Tympanuchus_phasianellus', u'Myiodynastes_luteiventris',
      u'Pycnonotus_jocosus', u'Somateria_spectabilis', u'Oceanodroma_homochroa', u'Perdix_perdix',
      u'Rhynchophanes_mccownii', u'Tympanuchus_cupido', u'Buteogallus_anthracinus', u'Oceanodroma_furcata',
      u'Peucedramus_taeniatus', u'Amazilia_violiceps', u'Icterus_graduacauda', u'Euplectes_franciscanus',
      u'Trogon_elegans', u'Contopus_pertinax', u'Charadrius_montanus', u'Vireo_atricapilla',
      u'Synthliboramphus_antiquus', u'Micrathene_whitneyi', u'Synthliboramphus_scrippsi',
      u'Chroicocephalus_ridibundus', u'Cypseloides_niger', u'Dendragapus_obscurus', u'Agapornis_roseicollis',
      u'Acanthis_hornemanni', u'Toxostoma_lecontei', u'Alle_alle', u'Aythya_fuligula', u'Cyrtonyx_montezumae',
      u'Pluvialis_fulva', u'Calidris_pugnax', u'Psittacara_holochlorus', u'Strix_occidentalis', u'Vireo_vicinior',
      u'Psiloscops_flammeolus', u'Leucosticte_australis', u'Oceanodroma_castro', u'Pterodroma_hasitata',
      u'Gymnogyps_californianus', u'Centrocercus_urophasianus', u'Porphyrio_porphyrio', u'Psittacara_mitratus',
      u'Toxostoma_bendirei', u'Onychoprion_fuscatus', u'Surnia_ulula', u'Leucosticte_atrata',
      u'Antrostomus_arizonae', u'Tyrannus_crassirostris', u'Megascops_trichopsis', u'Larus_livens',
      u'Vireo_altiloquus', u'Calothorax_lucifer', u'Sula_nebouxii', u'Strix_nebulosa', u'Laterallus_jamaicensis',
      u'Setophaga_kirtlandii', u'Pavo_cristatus', u'Calcarius_pictus', u'Branta_leucopsis', u'Puffinus_carneipes',
      u'Falcipennis_canadensis', u'Empidonax_fulvifrons', u'Streptopelia_chinensis', u'Onychoprion_anaethetus',
      u'Puffinus_tenuirostris', u'Falco_femoralis', u'Catharus_bicknelli', u'Thectocercus_acuticaudatus',
      u'Setophaga_pitiayumi', u'Tympanuchus_pallidicinctus', u'Icterus_pectoralis', u'Coturnicops_noveboracensis',
      u'Phoebastria_immutabilis', u'Vanellus_vanellus', u'Anser_brachyrhynchus', u'Calidris_ptilocnemis',
      u'Amazona_oratrix', u'Ammodramus_bairdii', u'Acridotheres_tristis', u'Poecile_sclateri',
      u'Oceanodroma_microsoma', u'Lagopus_leucura', u'Psittacula_krameri', u'Aix_galericulata', u'Vidua_macroura',
      u'Anous_stolidus', u'Uria_lomvia', u'Sporophila_torqueola', u'Sula_dactylatra', u'Polioptila_nigriceps',
      u'Phaethon_aethereus', u'Basileuterus_rufifrons', u'Limosa_lapponica', u'Amazona_finschi',
      u'Hylocharis_leucotis', u'Gavia_adamsii', u'Aphelocoma_insularis', u'Larus_schistisagus',
      u'Columbina_talpacoti', u'Tyrannus_savana', u'Falco_rusticolus', u'Coccyzus_minor', u'Oenanthe_oenanthe',
      u'Gracula_religiosa', u'Glaucidium_brasilianum', u'Turdus_rufopalliatus', u'Vireo_flavoviridis',
      u'Molothrus_bonariensis', u'Heliomaster_constantii', u'Pachyramphus_aglaiae', u'Synthliboramphus_craveri',
      u'Calidris_ferruginea', u'Amphispiza_quinquestriata', u'Patagioenas_flavirostris', u'Oreothlypis_crissalis',
      u'Rhodothraupis_celaeno', u'Chondrohierax_uncinatus', u'Aegolius_funereus', u'Thryophilus_sinaloa',
      u'Anthus_cervinus', u'Amazilia_beryllina', u'Calidris_acuminata', u'Gavia_arctica', u'Pelagodroma_marina',
      u'Myiarchus_sagrae', u'Pterodroma_ultima', u'Pterodroma_cookii', u'Anas_falcata', u'Chlidonias_hybrida',
      u'Limosa_limosa', u'Psilorhinus_morio', u'Myiarchus_nuttingi', u'Fringilla_montifringilla',
      u'Centrocercus_minimus', u'Brotogeris_versicolurus', u'Pagophila_eburnea', u'Antrostomus_ridgwayi',
      u'Piranga_bidentata', u'Amazona_autumnalis', u'Amazona_amazonica', u'Calidris_ruficollis',
      u'Chordeiles_gundlachii', u'Calocitta_colliei', u'Nomonyx_dominicus', u'Synthliboramphus_hypoleucus',
      u'Cygnus_atratus', u'Jacana_spinosa', u'Emberiza_rustica', u'Anas_querquedula', u'Phaethon_lepturus',
      u'Stercorarius_skua', u'Basileuterus_culicivorus', u'Pterodroma_arminjoniana', u'Chen_canagica',
      u'Larus_crassirostris', u'Chloroceryle_amazona', u'Icterus_wagleri', u'Grus_grus', u'Fratercula_corniculata',
      u'Icterus_pustulatus', u'Ara_ararauna', u'Crotophaga_ani', u'Carduelis_carduelis', u'Aethia_psittacula',
      u'Cyanocompsa_parellina', u'Rhodostethia_rosea', u'Motacilla_alba', u'Charadrius_mongolus',
      u'Anser_serrirostris', u'Ara_severus', u'Pterodroma_feae', u'Turdus_assimilis', u'Pterodroma_sandwichensis',
      u'Melopsittacus_undulatus', u'Aramides_axillaris', u'Tiaris_olivaceus', u'Spindalis_zena', u'Anas_bahamensis',
      u'Mimus_gundlachii', u'Charadrius_collaris', u'Cuculus_canorus', u'Vireo_crassirostris',
      u'Phoenicopterus_ruber', u'Egretta_garzetta', u'Grus_monacha', u'Pycnonotus_cafer', u'Emberiza_pusilla',
      u'Lanius_cristatus', u'Tringa_glareola', u'Turdus_pilaris', u'Anas_formosa', u'Tringa_erythropus',
      u'Anthus_hodgsoni', u'Phoebastria_albatrus', u'Fringilla_coelebs', u'Tringa_stagnatilis',
      u'Cygnus_melancoryphus', u'Charadrius_hiaticula', u'Calidris_minuta', u'Anser_fabalis', u'Amazona_aestiva',
      u'Lagopus_lagopus', u'Gallus_gallus', u'Sula_sula', u'Colibri_thalassinus', u'Spermestes_cucullata',
      u'Ridgwayia_pinicola', u'Falco_subbuteo', u'Tetraogallus_himalayensis', u'Melanotis_caerulescens',
      u'Myioborus_miniatus', u'Tadorna_tadorna', u'Anthracothorax_prevostii', u'Anous_minutus',
      u'Zosterops_japonicus', u'Chroicocephalus_cirrocephalus', u'Larus_dominicanus', u'Basileuterus_lachrymosus',
      u'Psittacara_leucophthalmus', u'Mergellus_albellus', u'Corvus_imparatus', u'Charadrius_leschenaultii',
      u'Geothlypis_poliocephala', u'Buteogallus_urubitinga', u'Geotrygon_chrysia', u'Egretta_gularis',
      u'Puffinus_baroli', u'Oreothlypis_superciliosa', u'Pterodroma_cahow', u'Cygnus_cygnus',
      u'Mitrephanes_phaeocercus', u'Pheucticus_chrysopeplus', u'Brachyramphus_perdix', u'Hydrobates_pelagicus',
      u'Tiaris_bicolor', u'Contopus_caribaeus', u'Pluvialis_apricaria', u'Legatus_leucophaius',
      u'Tigrisoma_mexicanum', u'Amazona_albifrons', u'Progne_tapera', u'Rupornis_magnirostris',
      u'Nymphicus_hollandicus', u'Agelaius_xanthomus', u'Motacilla_tschutschensis', u'Pionus_maximiliani',
      u'Streptopelia_roseogrisea', u'Psittacara_finschi', u'Chlidonias_leucopterus', u'Pterodroma_inexpectata',
      u'Cyanerpes_cyaneus', u'Coereba_flaveola', u'Alauda_arvensis', u'Agapornis_personatus', u'Psittacara_wagleri',
      u'Plectrophenax_hyperboreus', u'Euptilotis_neoxenus', u'Phylloscopus_fuscatus', u'Estrilda_melpoda',
      u'Phylloscopus_borealis', u'Margarops_fuscatus', u'Procellaria_aequinoctialis', u'Calonectris_leucomelas',
      u'Thalassarche_salvini', u'Parus_major', u'Tringa_brevipes', u'Oceanodroma_tethys',
      u'Catharus_aurantiirostris', u'Zenaida_aurita', u'Rissa_brevirostris', u'Falco_tinnunculus',
      u'Tachycineta_cyaneoviridis', u'Estrilda_caerulescens', u'Turdus_iliacus', u'Callonetta_leucophrys',
      u'Anser_anser', u'Myadestes_occidentalis', u'Polysticta_stelleri', u'Numida_meleagris',
      u'Amazona_ochrocephala', u'Pterodroma_macroptera', u'Charadrius_morinellus', u'Thalassarche_cauta',
      u'Falco_vespertinus', u'Numenius_minutus', u'Dendrocygna_viduata', u'Larus_belcheri', u'Lonchura_malacca',
      u'Numenius_tahitiensis', u'Thalassarche_melanophris', u'Calidris_temminckii', u'Calonectris_edwardsii',
      u'Anas_sibilatrix', u'Tyrannus_caudifasciatus', u'Aythya_ferina', u'Calliphlox_evelynae',
      u'Ptiliogonys_cinereus', u'Prunella_montanella', u'Empidonomus_varius', u'Jabiru_mycteria',
      u'Bulweria_bulwerii', u'Eudocimus_ruber', u'Larus_michahellis', u'Hylocharis_xantusii',
      u'Thalassarche_chlororhynchos', u'Calidris_falcinellus', u'Luscinia_svecica', u'Xenus_cinereus',
      u'Oceanodroma_monorhis', u'Pterodroma_heraldica', u'Myiozetetes_similis', u'Micrastur_semitorquatus',
      u'Tadorna_ferruginea', u'Fregata_ariel', u'Elaenia_albiceps', u'Ara_militaris', u'Catharus_mexicanus',
      u'Geranospiza_caerulescens', u'Chloris_sinica', u'Puffinus_pacificus', u'Vireo_magister',
      u'Ficedula_albicilla', u'Lophonetta_specularioides', u'Aratinga_weddellii', u'Amazona_farinosa',
      u'Diomedea_exulans', u'Pterodroma_madeira', u'Anthropoides_virgo', u'Tityra_semifasciata',
      u'Tachycineta_albilinea', u'Carpodacus_erythrinus', u'Heliornis_fulica', u'Pterodroma_longirostris',
      u'Turdus_plumbeus', u'Empidonomus_aurantioatrocristatus', u'Calidris_tenuirostris', u'Geotrygon_montana',
      u'Tringa_nebularia', u'Phaethon_rubricauda', u'Threskiornis_aethiopicus', u'Progne_dominicensis',
      u'Tachornis_phoenicobia', u'Patagioenas_squamosa', u'Calidris_subminuta', u'Campephilus_principalis',
      u'Rhynchopsitta_pachyrhyncha', u'Harpagus_bidentatus', u'Amazona_auropalliata', u'Creagrus_furcatus',
      u'Procellaria_parkinsoni', u'Uraeginthus_bengalus', u'Corvus_splendens', u'Numenius_arquata',
      u'Tiaris_canorus', u'Corvus_monedula', u'Ploceus_cucullatus', u'Puffinus_auricularis', u'Tarsiger_cyanurus',
      u'Spinus_spinus', u'SAMPLING_EVENT_ID', u'LOC_ID', u'POP00_SQMI', u'HOUSING_DENSITY',
      u'HOUSING_PERCENT_VACANT', u'ELEV_GT', u'ELEV_NED', u'BCR', u'BAILEY_ECOREGION', u'OMERNIK_L3_ECOREGION',
      u'CAUS_TEMP_AVG', u'CAUS_TEMP_MIN', u'CAUS_TEMP_MAX', u'CAUS_PREC', u'CAUS_SNOW',
      u'NLCD2001_FS_C11_7500_PLAND', u'NLCD2001_FS_C12_7500_PLAND', u'NLCD2001_FS_C21_7500_PLAND',
      u'NLCD2001_FS_C22_7500_PLAND', u'NLCD2001_FS_C23_7500_PLAND', u'NLCD2001_FS_C24_7500_PLAND',
      u'NLCD2001_FS_C31_7500_PLAND', u'NLCD2001_FS_C41_7500_PLAND', u'NLCD2001_FS_C42_7500_PLAND',
      u'NLCD2001_FS_C43_7500_PLAND', u'NLCD2001_FS_C52_7500_PLAND', u'NLCD2001_FS_C71_7500_PLAND',
      u'NLCD2001_FS_C81_7500_PLAND', u'NLCD2001_FS_C82_7500_PLAND', u'NLCD2001_FS_C90_7500_PLAND',
      u'NLCD2001_FS_C95_7500_PLAND', u'NLCD2006_FS_C11_7500_PLAND', u'NLCD2006_FS_C12_7500_PLAND',
      u'NLCD2006_FS_C21_7500_PLAND', u'NLCD2006_FS_C22_7500_PLAND', u'NLCD2006_FS_C23_7500_PLAND',
      u'NLCD2006_FS_C24_7500_PLAND', u'NLCD2006_FS_C31_7500_PLAND', u'NLCD2006_FS_C41_7500_PLAND',
      u'NLCD2006_FS_C42_7500_PLAND', u'NLCD2006_FS_C43_7500_PLAND', u'NLCD2006_FS_C52_7500_PLAND',
      u'NLCD2006_FS_C71_7500_PLAND', u'NLCD2006_FS_C81_7500_PLAND', u'NLCD2006_FS_C82_7500_PLAND',
      u'NLCD2006_FS_C90_7500_PLAND', u'NLCD2006_FS_C95_7500_PLAND', u'NLCD2011_FS_C11_7500_PLAND',
      u'NLCD2011_FS_C12_7500_PLAND', u'NLCD2011_FS_C21_7500_PLAND', u'NLCD2011_FS_C22_7500_PLAND',
      u'NLCD2011_FS_C23_7500_PLAND', u'NLCD2011_FS_C24_7500_PLAND', u'NLCD2011_FS_C31_7500_PLAND',
      u'NLCD2011_FS_C41_7500_PLAND', u'NLCD2011_FS_C42_7500_PLAND', u'NLCD2011_FS_C43_7500_PLAND',
      u'NLCD2011_FS_C52_7500_PLAND', u'NLCD2011_FS_C71_7500_PLAND', u'NLCD2011_FS_C81_7500_PLAND',
      u'NLCD2011_FS_C82_7500_PLAND', u'NLCD2011_FS_C90_7500_PLAND', u'NLCD2011_FS_C95_7500_PLAND',
      u'SAMPLING_EVENT_ID', u'LOC_ID', u'SUBNATIONAL2_CODE', u'CAUS_TEMP_AVG01', u'CAUS_TEMP_AVG02',
      u'CAUS_TEMP_AVG03', u'CAUS_TEMP_AVG04', u'CAUS_TEMP_AVG05', u'CAUS_TEMP_AVG06', u'CAUS_TEMP_AVG07',
      u'CAUS_TEMP_AVG08', u'CAUS_TEMP_AVG09', u'CAUS_TEMP_AVG10', u'CAUS_TEMP_AVG11', u'CAUS_TEMP_AVG12',
      u'CAUS_TEMP_AVG13', u'CAUS_TEMP_MIN01', u'CAUS_TEMP_MIN02', u'CAUS_TEMP_MIN03', u'CAUS_TEMP_MIN04',
      u'CAUS_TEMP_MIN05', u'CAUS_TEMP_MIN06', u'CAUS_TEMP_MIN07', u'CAUS_TEMP_MIN08', u'CAUS_TEMP_MIN09',
      u'CAUS_TEMP_MIN10', u'CAUS_TEMP_MIN11', u'CAUS_TEMP_MIN12', u'CAUS_TEMP_MIN13', u'CAUS_TEMP_MAX01',
      u'CAUS_TEMP_MAX02', u'CAUS_TEMP_MAX03', u'CAUS_TEMP_MAX04', u'CAUS_TEMP_MAX05', u'CAUS_TEMP_MAX06',
      u'CAUS_TEMP_MAX07', u'CAUS_TEMP_MAX08', u'CAUS_TEMP_MAX09', u'CAUS_TEMP_MAX10', u'CAUS_TEMP_MAX11',
      u'CAUS_TEMP_MAX12', u'CAUS_TEMP_MAX13', u'CAUS_PREC01', u'CAUS_PREC02', u'CAUS_PREC03', u'CAUS_PREC04',
      u'CAUS_PREC05', u'CAUS_PREC06', u'CAUS_PREC07', u'CAUS_PREC08', u'CAUS_PREC09', u'CAUS_PREC10', u'CAUS_PREC11',
      u'CAUS_PREC12', u'CAUS_PREC13', u'CAUS_SNOW01', u'CAUS_SNOW02', u'CAUS_SNOW03', u'CAUS_SNOW04', u'CAUS_SNOW10',
      u'CAUS_SNOW11', u'CAUS_SNOW12', u'CAUS_LAST_SPRING_32F_MEDIAN', u'CAUS_LAST_SPRING_32F_MEAN',
      u'CAUS_LAST_SPRING_32F_EXTREME', u'CAUS_FIRST_AUTUMN_32F_MEDIAN', u'CAUS_FIRST_AUTUMN_32F_MEAN',
      u'CAUS_FIRST_AUTUMN_32F_EXTREME', u'NLCD01_CANOPYMEAN_RAD75', u'NLCD01_CANOPYMEAN_RAD750',
      u'NLCD01_CANOPYMEAN_RAD7500', u'NLCD01_IMPERVMEAN_RAD75', u'NLCD01_IMPERVMEAN_RAD750',
      u'NLCD01_IMPERVMEAN_RAD7500', u'DIST_FROM_FLOWING_FRESH', u'DIST_IN_FLOWING_FRESH',
      u'DIST_FROM_STANDING_FRESH', u'DIST_IN_STANDING_FRESH', u'DIST_FROM_WET_VEG_FRESH', u'DIST_IN_WET_VEG_FRESH',
      u'DIST_FROM_FLOWING_BRACKISH', u'DIST_IN_FLOWING_BRACKISH', u'DIST_FROM_STANDING_BRACKISH',
      u'DIST_IN_STANDING_BRACKISH', u'DIST_FROM_WET_VEG_BRACKISH', u'DIST_IN_WET_VEG_BRACKISH',
      u'NLCD2001_FS_L_75_ED', u'NLCD2001_FS_L_75_LPI', u'NLCD2001_FS_L_75_PD', u'NLCD2001_FS_L_750_ED',
      u'NLCD2001_FS_L_750_LPI', u'NLCD2001_FS_L_750_PD', u'NLCD2001_FS_L_7500_ED', u'NLCD2001_FS_L_7500_LPI',
      u'NLCD2001_FS_L_7500_PD', u'NLCD2001_FS_C11_75_ED', u'NLCD2001_FS_C11_75_LPI', u'NLCD2001_FS_C11_75_PD',
      u'NLCD2001_FS_C11_75_PLAND', u'NLCD2001_FS_C12_75_ED', u'NLCD2001_FS_C12_75_LPI', u'NLCD2001_FS_C12_75_PD',
      u'NLCD2001_FS_C12_75_PLAND', u'NLCD2001_FS_C21_75_ED', u'NLCD2001_FS_C21_75_LPI', u'NLCD2001_FS_C21_75_PD',
      u'NLCD2001_FS_C21_75_PLAND', u'NLCD2001_FS_C22_75_ED', u'NLCD2001_FS_C22_75_LPI', u'NLCD2001_FS_C22_75_PD',
      u'NLCD2001_FS_C22_75_PLAND', u'NLCD2001_FS_C23_75_ED', u'NLCD2001_FS_C23_75_LPI', u'NLCD2001_FS_C23_75_PD',
      u'NLCD2001_FS_C23_75_PLAND', u'NLCD2001_FS_C24_75_ED', u'NLCD2001_FS_C24_75_LPI', u'NLCD2001_FS_C24_75_PD',
      u'NLCD2001_FS_C24_75_PLAND', u'NLCD2001_FS_C31_75_ED', u'NLCD2001_FS_C31_75_LPI', u'NLCD2001_FS_C31_75_PD',
      u'NLCD2001_FS_C31_75_PLAND', u'NLCD2001_FS_C41_75_ED', u'NLCD2001_FS_C41_75_LPI', u'NLCD2001_FS_C41_75_PD',
      u'NLCD2001_FS_C41_75_PLAND', u'NLCD2001_FS_C42_75_ED', u'NLCD2001_FS_C42_75_LPI', u'NLCD2001_FS_C42_75_PD',
      u'NLCD2001_FS_C42_75_PLAND', u'NLCD2001_FS_C43_75_ED', u'NLCD2001_FS_C43_75_LPI', u'NLCD2001_FS_C43_75_PD',
      u'NLCD2001_FS_C43_75_PLAND', u'NLCD2001_FS_C52_75_ED', u'NLCD2001_FS_C52_75_LPI', u'NLCD2001_FS_C52_75_PD',
      u'NLCD2001_FS_C52_75_PLAND', u'NLCD2001_FS_C71_75_ED', u'NLCD2001_FS_C71_75_LPI', u'NLCD2001_FS_C71_75_PD',
      u'NLCD2001_FS_C71_75_PLAND', u'NLCD2001_FS_C81_75_ED', u'NLCD2001_FS_C81_75_LPI', u'NLCD2001_FS_C81_75_PD',
      u'NLCD2001_FS_C81_75_PLAND', u'NLCD2001_FS_C82_75_ED', u'NLCD2001_FS_C82_75_LPI', u'NLCD2001_FS_C82_75_PD',
      u'NLCD2001_FS_C82_75_PLAND', u'NLCD2001_FS_C90_75_ED', u'NLCD2001_FS_C90_75_LPI', u'NLCD2001_FS_C90_75_PD',
      u'NLCD2001_FS_C90_75_PLAND', u'NLCD2001_FS_C95_75_ED', u'NLCD2001_FS_C95_75_LPI', u'NLCD2001_FS_C95_75_PD',
      u'NLCD2001_FS_C95_75_PLAND', u'NLCD2001_FS_C11_750_ED', u'NLCD2001_FS_C11_750_LPI', u'NLCD2001_FS_C11_750_PD',
      u'NLCD2001_FS_C11_750_PLAND', u'NLCD2001_FS_C12_750_ED', u'NLCD2001_FS_C12_750_LPI', u'NLCD2001_FS_C12_750_PD',
      u'NLCD2001_FS_C12_750_PLAND', u'NLCD2001_FS_C21_750_ED', u'NLCD2001_FS_C21_750_LPI', u'NLCD2001_FS_C21_750_PD',
      u'NLCD2001_FS_C21_750_PLAND', u'NLCD2001_FS_C22_750_ED', u'NLCD2001_FS_C22_750_LPI', u'NLCD2001_FS_C22_750_PD',
      u'NLCD2001_FS_C22_750_PLAND', u'NLCD2001_FS_C23_750_ED', u'NLCD2001_FS_C23_750_LPI', u'NLCD2001_FS_C23_750_PD',
      u'NLCD2001_FS_C23_750_PLAND', u'NLCD2001_FS_C24_750_ED', u'NLCD2001_FS_C24_750_LPI', u'NLCD2001_FS_C24_750_PD',
      u'NLCD2001_FS_C24_750_PLAND', u'NLCD2001_FS_C31_750_ED', u'NLCD2001_FS_C31_750_LPI', u'NLCD2001_FS_C31_750_PD',
      u'NLCD2001_FS_C31_750_PLAND', u'NLCD2001_FS_C41_750_ED', u'NLCD2001_FS_C41_750_LPI', u'NLCD2001_FS_C41_750_PD',
      u'NLCD2001_FS_C41_750_PLAND', u'NLCD2001_FS_C42_750_ED', u'NLCD2001_FS_C42_750_LPI', u'NLCD2001_FS_C42_750_PD',
      u'NLCD2001_FS_C42_750_PLAND', u'NLCD2001_FS_C43_750_ED', u'NLCD2001_FS_C43_750_LPI', u'NLCD2001_FS_C43_750_PD',
      u'NLCD2001_FS_C43_750_PLAND', u'NLCD2001_FS_C52_750_ED', u'NLCD2001_FS_C52_750_LPI', u'NLCD2001_FS_C52_750_PD',
      u'NLCD2001_FS_C52_750_PLAND', u'NLCD2001_FS_C71_750_ED', u'NLCD2001_FS_C71_750_LPI', u'NLCD2001_FS_C71_750_PD',
      u'NLCD2001_FS_C71_750_PLAND', u'NLCD2001_FS_C81_750_ED', u'NLCD2001_FS_C81_750_LPI', u'NLCD2001_FS_C81_750_PD',
      u'NLCD2001_FS_C81_750_PLAND', u'NLCD2001_FS_C82_750_ED', u'NLCD2001_FS_C82_750_LPI', u'NLCD2001_FS_C82_750_PD',
      u'NLCD2001_FS_C82_750_PLAND', u'NLCD2001_FS_C90_750_ED', u'NLCD2001_FS_C90_750_LPI', u'NLCD2001_FS_C90_750_PD',
      u'NLCD2001_FS_C90_750_PLAND', u'NLCD2001_FS_C95_750_ED', u'NLCD2001_FS_C95_750_LPI', u'NLCD2001_FS_C95_750_PD',
      u'NLCD2001_FS_C95_750_PLAND', u'NLCD2001_FS_C11_7500_ED', u'NLCD2001_FS_C11_7500_LPI',
      u'NLCD2001_FS_C11_7500_PD', u'NLCD2001_FS_C12_7500_ED', u'NLCD2001_FS_C12_7500_LPI',
      u'NLCD2001_FS_C12_7500_PD', u'NLCD2001_FS_C21_7500_ED', u'NLCD2001_FS_C21_7500_LPI',
      u'NLCD2001_FS_C21_7500_PD', u'NLCD2001_FS_C22_7500_ED', u'NLCD2001_FS_C22_7500_LPI',
      u'NLCD2001_FS_C22_7500_PD', u'NLCD2001_FS_C23_7500_ED', u'NLCD2001_FS_C23_7500_LPI',
      u'NLCD2001_FS_C23_7500_PD', u'NLCD2001_FS_C24_7500_ED', u'NLCD2001_FS_C24_7500_LPI',
      u'NLCD2001_FS_C24_7500_PD', u'NLCD2001_FS_C31_7500_ED', u'NLCD2001_FS_C31_7500_LPI',
      u'NLCD2001_FS_C31_7500_PD', u'NLCD2001_FS_C41_7500_ED', u'NLCD2001_FS_C41_7500_LPI',
      u'NLCD2001_FS_C41_7500_PD', u'NLCD2001_FS_C42_7500_ED', u'NLCD2001_FS_C42_7500_LPI',
      u'NLCD2001_FS_C42_7500_PD', u'NLCD2001_FS_C43_7500_ED', u'NLCD2001_FS_C43_7500_LPI',
      u'NLCD2001_FS_C43_7500_PD', u'NLCD2001_FS_C52_7500_ED', u'NLCD2001_FS_C52_7500_LPI',
      u'NLCD2001_FS_C52_7500_PD', u'NLCD2001_FS_C71_7500_ED', u'NLCD2001_FS_C71_7500_LPI',
      u'NLCD2001_FS_C71_7500_PD', u'NLCD2001_FS_C81_7500_ED', u'NLCD2001_FS_C81_7500_LPI',
      u'NLCD2001_FS_C81_7500_PD', u'NLCD2001_FS_C82_7500_ED', u'NLCD2001_FS_C82_7500_LPI',
      u'NLCD2001_FS_C82_7500_PD', u'NLCD2001_FS_C90_7500_ED', u'NLCD2001_FS_C90_7500_LPI',
      u'NLCD2001_FS_C90_7500_PD', u'NLCD2001_FS_C95_7500_ED', u'NLCD2001_FS_C95_7500_LPI',
      u'NLCD2001_FS_C95_7500_PD', u'NLCD2006_FS_L_75_ED', u'NLCD2006_FS_L_75_LPI', u'NLCD2006_FS_L_75_PD',
      u'NLCD2006_FS_L_750_ED', u'NLCD2006_FS_L_750_LPI', u'NLCD2006_FS_L_750_PD', u'NLCD2006_FS_L_7500_ED',
      u'NLCD2006_FS_L_7500_LPI', u'NLCD2006_FS_L_7500_PD', u'NLCD2006_FS_C11_75_ED', u'NLCD2006_FS_C11_75_LPI',
      u'NLCD2006_FS_C11_75_PD', u'NLCD2006_FS_C11_75_PLAND', u'NLCD2006_FS_C12_75_ED', u'NLCD2006_FS_C12_75_LPI',
      u'NLCD2006_FS_C12_75_PD', u'NLCD2006_FS_C12_75_PLAND', u'NLCD2006_FS_C21_75_ED', u'NLCD2006_FS_C21_75_LPI',
      u'NLCD2006_FS_C21_75_PD', u'NLCD2006_FS_C21_75_PLAND', u'NLCD2006_FS_C22_75_ED', u'NLCD2006_FS_C22_75_LPI',
      u'NLCD2006_FS_C22_75_PD', u'NLCD2006_FS_C22_75_PLAND', u'NLCD2006_FS_C23_75_ED', u'NLCD2006_FS_C23_75_LPI',
      u'NLCD2006_FS_C23_75_PD', u'NLCD2006_FS_C23_75_PLAND', u'NLCD2006_FS_C24_75_ED', u'NLCD2006_FS_C24_75_LPI',
      u'NLCD2006_FS_C24_75_PD', u'NLCD2006_FS_C24_75_PLAND', u'NLCD2006_FS_C31_75_ED', u'NLCD2006_FS_C31_75_LPI',
      u'NLCD2006_FS_C31_75_PD', u'NLCD2006_FS_C31_75_PLAND', u'NLCD2006_FS_C41_75_ED', u'NLCD2006_FS_C41_75_LPI',
      u'NLCD2006_FS_C41_75_PD', u'NLCD2006_FS_C41_75_PLAND', u'NLCD2006_FS_C42_75_ED', u'NLCD2006_FS_C42_75_LPI',
      u'NLCD2006_FS_C42_75_PD', u'NLCD2006_FS_C42_75_PLAND', u'NLCD2006_FS_C43_75_ED', u'NLCD2006_FS_C43_75_LPI',
      u'NLCD2006_FS_C43_75_PD', u'NLCD2006_FS_C43_75_PLAND', u'NLCD2006_FS_C52_75_ED', u'NLCD2006_FS_C52_75_LPI',
      u'NLCD2006_FS_C52_75_PD', u'NLCD2006_FS_C52_75_PLAND', u'NLCD2006_FS_C71_75_ED', u'NLCD2006_FS_C71_75_LPI',
      u'NLCD2006_FS_C71_75_PD', u'NLCD2006_FS_C71_75_PLAND', u'NLCD2006_FS_C81_75_ED', u'NLCD2006_FS_C81_75_LPI',
      u'NLCD2006_FS_C81_75_PD', u'NLCD2006_FS_C81_75_PLAND', u'NLCD2006_FS_C82_75_ED', u'NLCD2006_FS_C82_75_LPI',
      u'NLCD2006_FS_C82_75_PD', u'NLCD2006_FS_C82_75_PLAND', u'NLCD2006_FS_C90_75_ED', u'NLCD2006_FS_C90_75_LPI',
      u'NLCD2006_FS_C90_75_PD', u'NLCD2006_FS_C90_75_PLAND', u'NLCD2006_FS_C95_75_ED', u'NLCD2006_FS_C95_75_LPI',
      u'NLCD2006_FS_C95_75_PD', u'NLCD2006_FS_C95_75_PLAND', u'NLCD2006_FS_C11_750_ED', u'NLCD2006_FS_C11_750_LPI',
      u'NLCD2006_FS_C11_750_PD', u'NLCD2006_FS_C11_750_PLAND', u'NLCD2006_FS_C12_750_ED', u'NLCD2006_FS_C12_750_LPI',
      u'NLCD2006_FS_C12_750_PD', u'NLCD2006_FS_C12_750_PLAND', u'NLCD2006_FS_C21_750_ED', u'NLCD2006_FS_C21_750_LPI',
      u'NLCD2006_FS_C21_750_PD', u'NLCD2006_FS_C21_750_PLAND', u'NLCD2006_FS_C22_750_ED', u'NLCD2006_FS_C22_750_LPI',
      u'NLCD2006_FS_C22_750_PD', u'NLCD2006_FS_C22_750_PLAND', u'NLCD2006_FS_C23_750_ED', u'NLCD2006_FS_C23_750_LPI',
      u'NLCD2006_FS_C23_750_PD', u'NLCD2006_FS_C23_750_PLAND', u'NLCD2006_FS_C24_750_ED', u'NLCD2006_FS_C24_750_LPI',
      u'NLCD2006_FS_C24_750_PD', u'NLCD2006_FS_C24_750_PLAND', u'NLCD2006_FS_C31_750_ED', u'NLCD2006_FS_C31_750_LPI',
      u'NLCD2006_FS_C31_750_PD', u'NLCD2006_FS_C31_750_PLAND', u'NLCD2006_FS_C41_750_ED', u'NLCD2006_FS_C41_750_LPI',
      u'NLCD2006_FS_C41_750_PD', u'NLCD2006_FS_C41_750_PLAND', u'NLCD2006_FS_C42_750_ED', u'NLCD2006_FS_C42_750_LPI',
      u'NLCD2006_FS_C42_750_PD', u'NLCD2006_FS_C42_750_PLAND', u'NLCD2006_FS_C43_750_ED', u'NLCD2006_FS_C43_750_LPI',
      u'NLCD2006_FS_C43_750_PD', u'NLCD2006_FS_C43_750_PLAND', u'NLCD2006_FS_C52_750_ED', u'NLCD2006_FS_C52_750_LPI',
      u'NLCD2006_FS_C52_750_PD', u'NLCD2006_FS_C52_750_PLAND', u'NLCD2006_FS_C71_750_ED', u'NLCD2006_FS_C71_750_LPI',
      u'NLCD2006_FS_C71_750_PD', u'NLCD2006_FS_C71_750_PLAND', u'NLCD2006_FS_C81_750_ED', u'NLCD2006_FS_C81_750_LPI',
      u'NLCD2006_FS_C81_750_PD', u'NLCD2006_FS_C81_750_PLAND', u'NLCD2006_FS_C82_750_ED', u'NLCD2006_FS_C82_750_LPI',
      u'NLCD2006_FS_C82_750_PD', u'NLCD2006_FS_C82_750_PLAND', u'NLCD2006_FS_C90_750_ED', u'NLCD2006_FS_C90_750_LPI',
      u'NLCD2006_FS_C90_750_PD', u'NLCD2006_FS_C90_750_PLAND', u'NLCD2006_FS_C95_750_ED', u'NLCD2006_FS_C95_750_LPI',
      u'NLCD2006_FS_C95_750_PD', u'NLCD2006_FS_C95_750_PLAND', u'NLCD2006_FS_C11_7500_ED',
      u'NLCD2006_FS_C11_7500_LPI', u'NLCD2006_FS_C11_7500_PD', u'NLCD2006_FS_C12_7500_ED',
      u'NLCD2006_FS_C12_7500_LPI', u'NLCD2006_FS_C12_7500_PD', u'NLCD2006_FS_C21_7500_ED',
      u'NLCD2006_FS_C21_7500_LPI', u'NLCD2006_FS_C21_7500_PD', u'NLCD2006_FS_C22_7500_ED',
      u'NLCD2006_FS_C22_7500_LPI', u'NLCD2006_FS_C22_7500_PD', u'NLCD2006_FS_C23_7500_ED',
      u'NLCD2006_FS_C23_7500_LPI', u'NLCD2006_FS_C23_7500_PD', u'NLCD2006_FS_C24_7500_ED',
      u'NLCD2006_FS_C24_7500_LPI', u'NLCD2006_FS_C24_7500_PD', u'NLCD2006_FS_C31_7500_ED',
      u'NLCD2006_FS_C31_7500_LPI', u'NLCD2006_FS_C31_7500_PD', u'NLCD2006_FS_C41_7500_ED',
      u'NLCD2006_FS_C41_7500_LPI', u'NLCD2006_FS_C41_7500_PD', u'NLCD2006_FS_C42_7500_ED',
      u'NLCD2006_FS_C42_7500_LPI', u'NLCD2006_FS_C42_7500_PD', u'NLCD2006_FS_C43_7500_ED',
      u'NLCD2006_FS_C43_7500_LPI', u'NLCD2006_FS_C43_7500_PD', u'NLCD2006_FS_C52_7500_ED',
      u'NLCD2006_FS_C52_7500_LPI', u'NLCD2006_FS_C52_7500_PD', u'NLCD2006_FS_C71_7500_ED',
      u'NLCD2006_FS_C71_7500_LPI', u'NLCD2006_FS_C71_7500_PD', u'NLCD2006_FS_C81_7500_ED',
      u'NLCD2006_FS_C81_7500_LPI', u'NLCD2006_FS_C81_7500_PD', u'NLCD2006_FS_C82_7500_ED',
      u'NLCD2006_FS_C82_7500_LPI', u'NLCD2006_FS_C82_7500_PD', u'NLCD2006_FS_C90_7500_ED',
      u'NLCD2006_FS_C90_7500_LPI', u'NLCD2006_FS_C90_7500_PD', u'NLCD2006_FS_C95_7500_ED',
      u'NLCD2006_FS_C95_7500_LPI', u'NLCD2006_FS_C95_7500_PD', u'NLCD2011_FS_L_75_ED', u'NLCD2011_FS_L_75_LPI',
      u'NLCD2011_FS_L_75_PD', u'NLCD2011_FS_L_750_ED', u'NLCD2011_FS_L_750_LPI', u'NLCD2011_FS_L_750_PD',
      u'NLCD2011_FS_L_7500_ED', u'NLCD2011_FS_L_7500_LPI', u'NLCD2011_FS_L_7500_PD', u'NLCD2011_FS_C11_75_ED',
      u'NLCD2011_FS_C11_75_LPI', u'NLCD2011_FS_C11_75_PD', u'NLCD2011_FS_C11_75_PLAND', u'NLCD2011_FS_C12_75_ED',
      u'NLCD2011_FS_C12_75_LPI', u'NLCD2011_FS_C12_75_PD', u'NLCD2011_FS_C12_75_PLAND', u'NLCD2011_FS_C21_75_ED',
      u'NLCD2011_FS_C21_75_LPI', u'NLCD2011_FS_C21_75_PD', u'NLCD2011_FS_C21_75_PLAND', u'NLCD2011_FS_C22_75_ED',
      u'NLCD2011_FS_C22_75_LPI', u'NLCD2011_FS_C22_75_PD', u'NLCD2011_FS_C22_75_PLAND', u'NLCD2011_FS_C23_75_ED',
      u'NLCD2011_FS_C23_75_LPI', u'NLCD2011_FS_C23_75_PD', u'NLCD2011_FS_C23_75_PLAND', u'NLCD2011_FS_C24_75_ED',
      u'NLCD2011_FS_C24_75_LPI', u'NLCD2011_FS_C24_75_PD', u'NLCD2011_FS_C24_75_PLAND', u'NLCD2011_FS_C31_75_ED',
      u'NLCD2011_FS_C31_75_LPI', u'NLCD2011_FS_C31_75_PD', u'NLCD2011_FS_C31_75_PLAND', u'NLCD2011_FS_C41_75_ED',
      u'NLCD2011_FS_C41_75_LPI', u'NLCD2011_FS_C41_75_PD', u'NLCD2011_FS_C41_75_PLAND', u'NLCD2011_FS_C42_75_ED',
      u'NLCD2011_FS_C42_75_LPI', u'NLCD2011_FS_C42_75_PD', u'NLCD2011_FS_C42_75_PLAND', u'NLCD2011_FS_C43_75_ED',
      u'NLCD2011_FS_C43_75_LPI', u'NLCD2011_FS_C43_75_PD', u'NLCD2011_FS_C43_75_PLAND', u'NLCD2011_FS_C52_75_ED',
      u'NLCD2011_FS_C52_75_LPI', u'NLCD2011_FS_C52_75_PD', u'NLCD2011_FS_C52_75_PLAND', u'NLCD2011_FS_C71_75_ED',
      u'NLCD2011_FS_C71_75_LPI', u'NLCD2011_FS_C71_75_PD', u'NLCD2011_FS_C71_75_PLAND', u'NLCD2011_FS_C81_75_ED',
      u'NLCD2011_FS_C81_75_LPI', u'NLCD2011_FS_C81_75_PD', u'NLCD2011_FS_C81_75_PLAND', u'NLCD2011_FS_C82_75_ED',
      u'NLCD2011_FS_C82_75_LPI', u'NLCD2011_FS_C82_75_PD', u'NLCD2011_FS_C82_75_PLAND', u'NLCD2011_FS_C90_75_ED',
      u'NLCD2011_FS_C90_75_LPI', u'NLCD2011_FS_C90_75_PD', u'NLCD2011_FS_C90_75_PLAND', u'NLCD2011_FS_C95_75_ED',
      u'NLCD2011_FS_C95_75_LPI', u'NLCD2011_FS_C95_75_PD', u'NLCD2011_FS_C95_75_PLAND', u'NLCD2011_FS_C11_750_ED',
      u'NLCD2011_FS_C11_750_LPI', u'NLCD2011_FS_C11_750_PD', u'NLCD2011_FS_C11_750_PLAND', u'NLCD2011_FS_C12_750_ED',
      u'NLCD2011_FS_C12_750_LPI', u'NLCD2011_FS_C12_750_PD', u'NLCD2011_FS_C12_750_PLAND', u'NLCD2011_FS_C21_750_ED',
      u'NLCD2011_FS_C21_750_LPI', u'NLCD2011_FS_C21_750_PD', u'NLCD2011_FS_C21_750_PLAND', u'NLCD2011_FS_C22_750_ED',
      u'NLCD2011_FS_C22_750_LPI', u'NLCD2011_FS_C22_750_PD', u'NLCD2011_FS_C22_750_PLAND', u'NLCD2011_FS_C23_750_ED',
      u'NLCD2011_FS_C23_750_LPI', u'NLCD2011_FS_C23_750_PD', u'NLCD2011_FS_C23_750_PLAND', u'NLCD2011_FS_C24_750_ED',
      u'NLCD2011_FS_C24_750_LPI', u'NLCD2011_FS_C24_750_PD', u'NLCD2011_FS_C24_750_PLAND', u'NLCD2011_FS_C31_750_ED',
      u'NLCD2011_FS_C31_750_LPI', u'NLCD2011_FS_C31_750_PD', u'NLCD2011_FS_C31_750_PLAND', u'NLCD2011_FS_C41_750_ED',
      u'NLCD2011_FS_C41_750_LPI', u'NLCD2011_FS_C41_750_PD', u'NLCD2011_FS_C41_750_PLAND', u'NLCD2011_FS_C42_750_ED',
      u'NLCD2011_FS_C42_750_LPI', u'NLCD2011_FS_C42_750_PD', u'NLCD2011_FS_C42_750_PLAND', u'NLCD2011_FS_C43_750_ED',
      u'NLCD2011_FS_C43_750_LPI', u'NLCD2011_FS_C43_750_PD', u'NLCD2011_FS_C43_750_PLAND', u'NLCD2011_FS_C52_750_ED',
      u'NLCD2011_FS_C52_750_LPI', u'NLCD2011_FS_C52_750_PD', u'NLCD2011_FS_C52_750_PLAND', u'NLCD2011_FS_C71_750_ED',
      u'NLCD2011_FS_C71_750_LPI', u'NLCD2011_FS_C71_750_PD', u'NLCD2011_FS_C71_750_PLAND', u'NLCD2011_FS_C81_750_ED',
      u'NLCD2011_FS_C81_750_LPI', u'NLCD2011_FS_C81_750_PD', u'NLCD2011_FS_C81_750_PLAND', u'NLCD2011_FS_C82_750_ED',
      u'NLCD2011_FS_C82_750_LPI', u'NLCD2011_FS_C82_750_PD', u'NLCD2011_FS_C82_750_PLAND', u'NLCD2011_FS_C90_750_ED',
      u'NLCD2011_FS_C90_750_LPI', u'NLCD2011_FS_C90_750_PD', u'NLCD2011_FS_C90_750_PLAND', u'NLCD2011_FS_C95_750_ED',
      u'NLCD2011_FS_C95_750_LPI', u'NLCD2011_FS_C95_750_PD', u'NLCD2011_FS_C95_750_PLAND',
      u'NLCD2011_FS_C11_7500_ED', u'NLCD2011_FS_C11_7500_LPI', u'NLCD2011_FS_C11_7500_PD',
      u'NLCD2011_FS_C12_7500_ED', u'NLCD2011_FS_C12_7500_LPI', u'NLCD2011_FS_C12_7500_PD',
      u'NLCD2011_FS_C21_7500_ED', u'NLCD2011_FS_C21_7500_LPI', u'NLCD2011_FS_C21_7500_PD',
      u'NLCD2011_FS_C22_7500_ED', u'NLCD2011_FS_C22_7500_LPI', u'NLCD2011_FS_C22_7500_PD',
      u'NLCD2011_FS_C23_7500_ED', u'NLCD2011_FS_C23_7500_LPI', u'NLCD2011_FS_C23_7500_PD',
      u'NLCD2011_FS_C24_7500_ED', u'NLCD2011_FS_C24_7500_LPI', u'NLCD2011_FS_C24_7500_PD',
      u'NLCD2011_FS_C31_7500_ED', u'NLCD2011_FS_C31_7500_LPI', u'NLCD2011_FS_C31_7500_PD',
      u'NLCD2011_FS_C41_7500_ED', u'NLCD2011_FS_C41_7500_LPI', u'NLCD2011_FS_C41_7500_PD',
      u'NLCD2011_FS_C42_7500_ED', u'NLCD2011_FS_C42_7500_LPI', u'NLCD2011_FS_C42_7500_PD',
      u'NLCD2011_FS_C43_7500_ED', u'NLCD2011_FS_C43_7500_LPI', u'NLCD2011_FS_C43_7500_PD',
      u'NLCD2011_FS_C52_7500_ED', u'NLCD2011_FS_C52_7500_LPI', u'NLCD2011_FS_C52_7500_PD',
      u'NLCD2011_FS_C71_7500_ED', u'NLCD2011_FS_C71_7500_LPI', u'NLCD2011_FS_C71_7500_PD',
      u'NLCD2011_FS_C81_7500_ED', u'NLCD2011_FS_C81_7500_LPI', u'NLCD2011_FS_C81_7500_PD',
      u'NLCD2011_FS_C82_7500_ED', u'NLCD2011_FS_C82_7500_LPI', u'NLCD2011_FS_C82_7500_PD',
      u'NLCD2011_FS_C90_7500_ED', u'NLCD2011_FS_C90_7500_LPI', u'NLCD2011_FS_C90_7500_PD',
      u'NLCD2011_FS_C95_7500_ED', u'NLCD2011_FS_C95_7500_LPI', u'NLCD2011_FS_C95_7500_PD'])
    birds_columns = DataExploration.cal_birds_column_ids(header_dict)
    drop_index = DataExploration.cal_drop_column_ids(header_dict)

    data_prediction.test_lr_prediction_through_models(data_set, model_path, prediction_path,
                                                   header_dict, birds_columns, drop_index)
    data_prediction.sc.stop()