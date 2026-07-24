#!/bin/bash
echo "▶️ [Worker 0] Старт: 90:15 фолд 2010..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2010-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2010-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2010.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2010"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2011..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2011-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2011-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2011.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2011"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2012..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2012-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2012-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2012.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2012"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2013..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2013-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2013-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2013.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2013"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2014..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2014-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2014-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2014.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2014"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2015..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2015-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2015-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2015.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2015"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2016..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2016-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2016-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2016.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2016"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2017..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2017-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2017-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2017.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2017"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2018..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2018-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2018-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2018.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2018"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2019..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2019-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2019-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2019.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2019"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2020..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2020-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2020-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2020.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2020"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2021..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2021-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2021-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2021.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2021"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2022..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2022-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2022-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2022.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2022"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2023..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2023-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2023-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2023.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2023"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2024..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2024-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2024-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2024.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2024"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2025..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2025-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2025-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2025.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2025"
echo "▶️ [Worker 0] Старт: 90:15 фолд 2026..."
python -m _tools.init_dataset \
    --timeframe 1d \
    --lookback 90 \
    --horizon 15 \
    --auto --percentile 75 \
    --init_split "2026-01-01" \
    --val_interval 2 \
    --split_interval 1 \
    --endpoint "2026-01-01" \
    --corr_threshold 0.85 \
    --cum_threshold 0.99 \
    --workers 1 > "_logs_data_prep/fold_90_15_2026.log" 2>&1
echo "✅ [Worker 0] Завершено: 90:15 фолд 2026"
