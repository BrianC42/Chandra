'''
Created on Apr 13, 2020

@author: Brian
'''
import sys
import logging
import networkx as nx
import configuration_constants as cc

def add_data_flow_details(js_flow_conditional, nx_graph, nx_edge_key):

    for edge_i in nx_graph.edges():
        if edge_i == (nx_edge_key[0], nx_edge_key[1]):
            if cc.JSON_TENSORFLOW_DATA in js_flow_conditional:
                js_tensorflow_data = js_flow_conditional[cc.JSON_TENSORFLOW_DATA]
                
                if cc.JSON_FEATURE_FIELDS in js_tensorflow_data:
                    nx_feature_fields = js_tensorflow_data[cc.JSON_FEATURE_FIELDS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_feature_fields}, cc.JSON_FEATURE_FIELDS)

                if cc.JSON_TARGET_FIELDS in js_tensorflow_data:
                    nx_target_fields = js_tensorflow_data[cc.JSON_TARGET_FIELDS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_target_fields}, cc.JSON_TARGET_FIELDS)

                if cc.JSON_TIME_SEQ in js_tensorflow_data:
                    nx_time_seq = js_tensorflow_data[cc.JSON_TIME_SEQ]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_time_seq}, cc.JSON_TIME_SEQ)

                if nx_time_seq:
                    nx_seriesStepID = js_tensorflow_data[cc.JSON_SERIES_ID]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_seriesStepID}, cc.JSON_SERIES_ID)

                    nx_seriesDataType = js_tensorflow_data[cc.JSON_SERIES_DATA_TYPE]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_seriesDataType}, cc.JSON_SERIES_DATA_TYPE)

    return 

def add_data_source_details(js_config, nx_graph, nx_process_name):
    nx_inputFile = js_config[cc.JSON_INPUT_DATA_FILE]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_inputFile}, cc.JSON_INPUT_DATA_FILE)
    return

def add_meta_data_edge (js_config, d2r):
    nx_flowName = js_config[cc.JSON_FLOW_NAME]

    # mandatory elements
    js_required = js_config[cc.JSON_REQUIRED]
    nx_flowFrom =  js_required[cc.JSON_FLOW_FROM]
    nx_flowTo =  js_required[cc.JSON_FLOW_TO]
    d2r.graph.add_edge(nx_flowFrom, nx_flowTo, key=nx_flowName)

    nx_edge_key = (nx_flowFrom, nx_flowTo, nx_flowName)
    nx_flow_data_file = js_required[cc.JSON_FLOW_DATA_FILE]
    nx.set_edge_attributes(d2r.graph, {nx_edge_key:nx_flow_data_file}, cc.JSON_FLOW_DATA_FILE)

    if cc.JSON_CONDITIONAL in js_config:
        js_conditional = js_config[cc.JSON_CONDITIONAL]
        add_data_flow_details(js_conditional, d2r.graph, nx_edge_key)

    print("Connected %s to %s by data flow %s" % (nx_flowFrom, nx_flowTo, nx_flowName))
    return

def add_meta_process_node (js_config, d2r) :
    nx_process_name = ""
    if cc.JSON_NODE_NAME in js_config:
        nx_process_name = js_config[cc.JSON_NODE_NAME]
    else:
        raise NameError('Missing process node name')
        
    d2r.graph.add_node(nx_process_name)

    # mandatory elements
    js_required = js_config[cc.JSON_REQUIRED]
    nx_processType = js_required[cc.JSON_PROCESS_TYPE]
    nx_inputs = js_required[cc.JSON_INPUT_FLOWS]
    nx_output = js_required[cc.JSON_OUTPUT_FLOW]
    #nx_outputFile = js_required[cc.JSON_OUTPUT_FILE]
    #nx_log = js_required[cc.JSON_LOG_FILE]
    
    for node_i in d2r.graph.nodes():
        if node_i == nx_process_name:
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_processType}, cc.JSON_PROCESS_TYPE)
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_inputs}, cc.JSON_INPUT_FLOWS)
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_output}, cc.JSON_OUTPUT_FLOW)
            #nx.set_node_attributes(nx_graph, {nx_process_name:nx_outputFile}, cc.JSON_OUTPUT_FILE)
            #nx.set_node_attributes(d2r.graph, {nx_process_name:nx_log}, cc.JSON_LOG_FILE)
            break
    
    # conditional elements
    js_conditional = js_config[cc.JSON_CONDITIONAL]
    
    if nx_processType == cc.JSON_DATA_LOAD_PROCESS:
        js_data_prep_ctrl = js_conditional[cc.JSON_INPUT_DATA_PREPARATION]
        add_data_source_details(js_data_prep_ctrl, d2r.graph, nx_process_name)
        
    elif nx_processType == cc.JSON_DATA_PREP_PROCESS:
        nx_prepCtrl = js_conditional[cc.JSON_DATA_PREPARATION_CTRL]
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_prepCtrl}, cc.JSON_DATA_PREPARATION_CTRL)
        js_data_prep_ctrl = js_conditional[cc.JSON_DATA_PREPARATION_CTRL]

        if cc.JSON_IGNORE_BLANKS in js_data_prep_ctrl:
            nx_ignore_blanks = js_data_prep_ctrl[cc.JSON_IGNORE_BLANKS]
        else:
            nx_ignore_blanks = False
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_ignore_blanks}, cc.JSON_IGNORE_BLANKS)

        '''
        add_data_prep_details(js_data_prep_ctrl, nx_graph, nx_process_name)
        '''

    elif nx_processType == cc.JSON_TENSORFLOW:
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_processType}, cc.JSON_TRAIN)
        nx_dataPrecision = js_conditional[cc.JSON_PRECISION]
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_dataPrecision}, cc.JSON_PRECISION)
        
        if cc.JSON_BALANCED in js_conditional:
            nx_balanced = js_conditional[cc.JSON_BALANCED]
        else:
            nx_balanced = False
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_balanced}, cc.JSON_BALANCED)

        if cc.JSON_ML_GOAL_COMBINE_SAMPLE_COUNT in js_conditional:
            nx_ml_goal_combine_sample_count = js_conditional[cc.JSON_ML_GOAL_COMBINE_SAMPLE_COUNT]
        else:
            nx_ml_goal_combine_sample_count = 1
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_ml_goal_combine_sample_count}, cc.JSON_ML_GOAL_COMBINE_SAMPLE_COUNT)
                     
        nx_visualizations = js_conditional[cc.JSON_VISUALIZATIONS]
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_visualizations}, cc.JSON_VISUALIZATIONS)
        
        if cc.JSON_NORMALIZE_DATA in js_conditional:
            nx_normalize = js_conditional[cc.JSON_NORMALIZE_DATA]
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_normalize}, cc.JSON_NORMALIZE_DATA)
        
        if cc.JSON_TENSORBOARD in js_conditional:
            nx_tensorboard = js_conditional[cc.JSON_TENSORBOARD]
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_tensorboard}, cc.JSON_TENSORBOARD)
            
        if cc.JSON_TRAINING_ITERATIONS in js_conditional:
            nx_trainingIteration = js_conditional[cc.JSON_TRAINING_ITERATIONS]
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_trainingIteration}, cc.JSON_TRAINING_ITERATIONS)
            d2r.trainingIterationCount = len(nx_trainingIteration)
            #print ("training iterations")

        if cc.JSON_ML_GOAL  in js_conditional:
            nx_ml_goal = js_conditional[cc.JSON_ML_GOAL]
            if nx_ml_goal == cc.JSON_ML_GOAL_CATEGORIZATION:
                pass
            elif  nx_ml_goal == cc.JSON_ML_GOAL_REGRESSION:
                pass
            else:
                raise NameError(nx_process_name + " invalid " + cc.JSON_ML_GOAL)
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_ml_goal}, cc.JSON_ML_GOAL)
        else:
            raise NameError(nx_process_name + " requires " + cc.JSON_ML_GOAL)
            
        if cc.JSON_ML_REGRESSION_FORECAST_INTERVAL in js_conditional:
            nx_regression_forecast_interval = js_conditional[cc.JSON_ML_REGRESSION_FORECAST_INTERVAL]
        else:
            nx_regression_forecast_interval = 1
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_regression_forecast_interval}, cc.JSON_ML_REGRESSION_FORECAST_INTERVAL)
             
    elif nx_processType == cc.JSON_EXECUTE_MODEL:
        pass

    elif nx_processType == cc.JSON_AUTOKERAS:
        print("\n============== WIP =============\n\tAdding auto Keras node details to processing network\n")
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_processType}, cc.JSON_TRAIN)
        nx_dataPrecision = js_conditional[cc.JSON_PRECISION]
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_dataPrecision}, cc.JSON_PRECISION)
        nx_visualizations = js_conditional[cc.JSON_VISUALIZATIONS]
        nx.set_node_attributes(d2r.graph, {nx_process_name:nx_visualizations}, cc.JSON_VISUALIZATIONS)

        if cc.JSON_AUTOKERAS_PARAMETERS in js_conditional:
            nx_akParameters = js_conditional[cc.JSON_AUTOKERAS_PARAMETERS]
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_akParameters}, cc.JSON_AUTOKERAS_PARAMETERS)
            
        if cc.JSON_NORMALIZE_DATA in js_conditional:
            nx_normalize = js_conditional[cc.JSON_NORMALIZE_DATA]
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_normalize}, cc.JSON_NORMALIZE_DATA)
        
        if cc.JSON_LAYERS in js_conditional:
            nx_layers = js_conditional[cc.JSON_LAYERS]
            nx.set_node_attributes(d2r.graph, {nx_process_name:nx_layers}, cc.JSON_LAYERS)
            
    elif nx_processType == cc.JSON_STOP:
        pass
        
    else:
        ex_txt = nx_process_name + ", type " + nx_processType + " is not supported"
        raise NameError(ex_txt)
    
    nx_read_attr = nx.get_node_attributes(d2r.graph, cc.JSON_PROCESS_TYPE)
    print("Added %s node %s" % (nx_read_attr[nx_process_name], nx_process_name))
    return

def build_configuration_graph(d2r, JSON_config):
    # error handling
    try:
        d2r.graph = nx.MultiDiGraph()
    
        # processing nodes                
        logging.debug("Adding processing nodes")
        exc_txt = "json configuration file error\n\tprocess node "
        for i, process_i in enumerate(JSON_config[cc.JSON_PROCESS_NODES]):
            add_meta_process_node(process_i, d2r)
        
        # processing data flows                
        logging.debug("Adding data flows")
        exc_txt = "json configuration file error\n\tdata flow "
        for i, flow_i in enumerate(JSON_config[cc.JSON_DATA_FLOWS]):
            add_meta_data_edge(flow_i, d2r)
        
        d2r.plotGraph()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + " " + exc_str
        logging.debug(exc_txt)
        sys.exit(exc_txt)

    return