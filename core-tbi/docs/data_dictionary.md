# Data dictionary

Feature tables require `dataset_id`, `animal_id`, `session_id`, `timepoint`, and `condition`. Pose tables use long format: `dataset_id`, `animal_id`, `session_id`, `video_id`, `frame`, `time_seconds`, `bodypart`, `x`, `y`, `z`, `likelihood`, and available experimental metadata. `likelihood` must be retained, including missing values.
