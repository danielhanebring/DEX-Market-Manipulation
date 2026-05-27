# Project Workflow

Uniswap V3 Subgraph ->

1. Ingestion - Get raw swaps
2. Preprocessing - Clean, cast and sort swaps
3. Feature Engineering - Create event features
4. Weak labeling - Create sandwich labels
5. Train event level models (LR, RF, IF, XGB) and Temporal models (LSTM, LSTM Autoencoder)
6. Evaluation
7. Results

# Folders

configs/ - Configuration for data collection
data/raw/ - Raw data from Uniswap V3
data/processed/ - Cleaned and sorted swap data
data/features/ - Event and sequence features
data/labels/ - Weak labels and external sets
src/ingestion/ - Code to collect data
src/preprocessing/ - Code for cleaning and ordering data
src/features/ - Code for feature engineering
src/labeling/ - Code for heuristic labeling
src/models/ - Implementation of models
src/evaluation/ - Metrics and time based evaluation
src/analysis/ - Jaredfromsubway.eth analysis
scripts/ - Runable pipelines and experiemnts
outputs/metrics/ - Model metrics
outputs/predictions/ - Model scores
outputs/reports/ - Trained models, rapports and csv

# Config

configs/data.yamll

Specifies:
* Datasource
* API endpoint
* Amount of swaps per GraphQL-question
* Max amount of pages that can be fetched
* Pool address
* Time period
* Output for the raw and processed data

# Ingestion

scripts/run_ingestion.py
* Reads enviroment variables and config
* Gets API endpoint
* Reads chosen pool and time period
* Creates a Subgraphclient
* Gets all swaps
* Saved result
* Saved metadata

Other ingestion components:
* subgraph_client.py - Creates client around HTTP call to GraphQL endpoint

* queries.py - Builds the graphQL queries

* fetch_swaps.py - Flattens returned JSON

# Preprocessing

scripts/run_preprocessing.py
Reads raw data och produces clean and sorted swap table

Components:
* build_processed_tables.py - Reads raw parquet files, concatenates them, cleans data, sorts and saves result

* clean_swaps.py - Also processing. Removes doubles and type casting

* ordering.py - Sorts swaps

# Feature engineering

scripts/run_feature_engineering.py
Runs event feature pipeline and saves feature table and saved metadata

* event_features.py - Creates features
    features:
        - Trade features
        swap_size_token features - Abs size of token
        trade_direction - What token goes in and what goes out
        tick_change_from_previous - Tick difference to prev swap

        - Price features
        abs_tick_change - Abs tick change

        sqrt_price_x96 - Poolprice

        sqrt_price_change_from_previous - Price change compared to prev swap

        relative_sqrt_price_change - Relativ price change

        - Timing features

        time_since_last_swap_in_pool - Seconds since prev swap

        same_block_event_count - Nr of swaps in same block

        same_block_pattern_flag - Flag if the block contains multiple swaps

        local_event_density_10 - Local activity in the 10 last events

        - Gas features

        gas_price_gwei - Gaspris in gwei (gwei = unit for gas price)

        gas_price_local_mean_20
        gas_price_local_median_20
        gas_price_relative_to_local_mean
        gas_price_relative_to_local_median
        gas_spike_flag - Flag for heavily increased gas

        - Address features

        same_sender_recent_count_20 - How often have the sender recently been seen?

        same_recipient_recent_count_20 - How often has the recipient been seen?

        sender_recipient_pair_recent_count_20 - Repetition of sender and recipient pairs?

        same_sender_same_block_count - Nr of events from same sender in the block

        same_recipient_same_block_count	

        - Sandwich specific features
        Here for every event the module examines previous event -> current event -> next event (The current event is regarded as the victim)

        position_in_block

        block_event_count - Nr of events in pool

        prev_sender_address - Sender before event
        next_sender_address - After event
        prev_origin_address - Origin before event
        next_origin_address - After event

        same_sender_before_after_flag - Flag if sender is before and after middle

        different_middle_sender_from_neighbors_flag - Middle is a different actor

        same_origin_before_after_flag - Same origin before and after

        three_event_pattern_indicator - Flag for the actual sandwich strcuture

        Ex:
        Swap 1 sender = Bot
        Swap 2 sender = Victim
        Swap 3 sender = Bot

        same_sender_before_after_flag = 1
        different_middle_sender_from_neighbors_flag = 1

        - Tick reversal

        prev_tick - Tick(price) befire middle

        next_tick - Tick after middle

        tick_change_before - Movement before middle

        tick_change_after - Movement after middle

        reversal_pattern_flag - If the price direction reverses

        combined_reversal_magnitude - Strength of the reversal

        Ex:
        tick: 100 -> 112 -> 101

        tick_change_before = +12
        tick_change_after = -11
        reversal_pattern_flag = 1

    - Trading sizes

    attacker_average_size_token0 - Avg size before/after middle

    relative_trade_size_token0 - Middle size relative to the neighbors

    attacker_vs_victim_size_ratio_token0 - Neighbors size relative to middle

    - Gas relative features

    gas_price_relative_to_block_mean
    gas_price_relative_to_block_median
    gas_price_relative_to_neighbors_mean

    - Combined features

    high_block_gas_context_flag - HIgh gas in the block

    high_relative_trade_size_flag - Big trading relative to the context

    strict_sandwich_support_flag - Does it follow the sandwich structure? Bot-Victim-Bot

# Weak labeling

scripts/run_sandwich_labeling.py
* Reads event features
* Defubes config for sandwich rile
* Identifies sequences that are candidates for the label
* Creates labels on event and sequencial level
* Saves label tables

For every block windows containing 3 swaps are created:
transaction1, transaction2, transaction3

Every triple is analysed by the following rules:
* transaction1 and transaction3 is the same sender (possible attacker)

* Transaction2 is a different sender (possible victim)

* Tick reversal - Price moves and then reverses

* Attacker size - Attacker transaction exists and are sufficiently big

* Gas pattern - Extra confident signal, elevated gas usually means that someone wants to get prioritized

# Sequence labels

Every triple is either classified as:
* Weak_anomaly - Follows the sandwich structure rule
* Suspicious - Follows parts of the rules
* Normal - No indication of sandwich attack

# Event labels

Every event that is in a tirple inheric from the related sequence. So if the event is not in a sequence that is weak_anomaly or suspicious the event is classified as normal.

# Event modeling

The prepare_random_forest_dataset() is used for most event level models.

This function joins features and labels with swap_id.
Only keeps normal and weak_anomaly (Because we want a binary classification problem)
Maps labels to binary target
Sorts the data

Eventmodels use:
70% Training data
15% Validation data
15% Test data

# Logistic Regression

src/models/baselines/logistic_regression_model.py

Logistic Regression is a simple linear baseline mode that tests if a linear combination of features can separate normal from weak_anomaly.

outputs/metrics/logistic_regression_metrics.json

# Random Forest

Random Forest builds many decision trees and combines their output. This is a good model here because sandwich attacks are built by combinations of high reversal, same addresses, high gas.

These non-linear combinations fit for the tree models.

outputs/metrics/random_forest_metrics.json

# XGBoost

XGBoost build decision trees sequencially, where every tree tires to improve former erros.

This was the strongest event level model.

outputs/metrics/xgboost_metrics.json

# Isolation Forest

Unsupervised anomaly detector

It is only trained on normal events and the idea is for it to give high scores to unusual events (sandwich attacks)

# Leakage control

Labels are created from sandwich rules and multiple features discribe almost the same rule.

This can introduce circular reasoning

The solution for this was no_leakage.

No leakage removes features that are closely related to the label definition:
same_block_pattern_flag
same_origin_before_after_flag
three_event_pattern_indicator
strict_sandwich_support_flag
sandwich_support_score
same_sender_before_after_flag
different_middle_sender_from_neighbors_flag
reversal_pattern_flag
tick_change_before
tick_change_after
combined_reversal_magnitude
relative_trade_size_token0
attacker_vs_victim_size_ratio_token0
relative_trade_size_token1
attacker_vs_victim_size_ratio_token1
high_block_gas_context_flag
high_relative_trade_size_flag

no_leakage better shows how well the models can generalize from less frequent signals.

# Sequences

Event level models sees every event as separate. To catch temporal behaviors sequences were created.

A sliding window in the same pool is used.

Event 1-20   -> Sequence 1
Event 2-21   -> Sequence 2
Event 3-22   -> Sequence 3
...

Every event in a sequence is represented by 12 features:
swap_size_token0
swap_size_token1
interarrival_seconds
same_block_event_count
local_event_density_10
gas_price_gwei
gas_price_relative_to_local_median
same_sender_recent_count_20
same_recipient_recent_count_20
sender_recipient_pair_recent_count_20
tick_change_from_previous
abs_tick_change

One sequence has 20 time points and 12 features.

If at least one event in the window has the label weak_anomaly the whole window is seen as weak_anomaly.

# LSTM Classifier

LSTM is a temporal model that reads a sequence of swaps and tries to predict if the sequence contains a weak_anomaly event.

Multiple experiments were conducted.

with_jared - Jaredfromsubway.eth events are in the training data
without_jared - Jared events excluded from training
full - All features used
no_block_context - Block context features removed
no_rule	- Rule alike features removed
no_leakage - strict leakage related features removed

# LSTM Autoencoder

Autoencoder is trained to reconstruct normal sequences. If the reconstruction error is high the sequence is regarded as deviating.

This is unsupervised sequence alternative to LSTM. The model is trained wthout weak_labels as target but weak_lables can be used after to measure how te anomaly scores relate to the labeled patern.

# Jaredfromsubway.eth analysis

The project used a known MEV related address for analysis 

Jaredfromsubway.eth : 0xAE2Fc483527B8EF99EB5D9B44875F005ba1FaE13

The purpose was to find swaps where Jared occurs as sender, recipient or origin and to identify strict sandwich like triples connected to the address,

What is found:
The address mainly occurs in origin_address and not just sender_address.

1728 Jared related events were found
245 strict sandwich like triples were found

These cases were used to test models that have been trained on weak labels to see if they can identifiy structures connected to the well known MEV actor.

Another independent evaluation was done.

Instead of labeling jared events, i let the model score all Jared events.

This does not introduce circular reasoning.

The top ranked Jared events were manually reviewed and did indeed have strong indications of manipulation.
