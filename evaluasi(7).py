def evaluate_precision_recall_f1(predictions, ground_truth):
    true_positives = sum(predictions[i] == ground_truth[i] == 1 for i in range(len(predictions)))
    false_positives = sum(predictions[i] == 1 and ground_truth[i] == 0 for i in range(len(predictions)))
    false_negatives = sum(predictions[i] == 0 and ground_truth[i] == 1 for i in range(len(predictions)))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

# Assume we have lists of predicted labels and ground truth labels
# predictions = [1, 0, 1, 0, ...]  # 1 for relevant, 0 for not relevant
# ground_truth = [1, 1, 0, 1, ...]  # 1 for relevant, 0 for not relevant
# Contoh prediksi dan kebenaran sejati
predictions = [1, 0, 1, 0, 1, 1]  # 1 untuk relevan, 0 untuk tidak relevan
ground_truth = [1, 1, 0, 1, 0, 1]  # 1 untuk relevan, 0 untuk tidak relevan

# Evaluasi
precision, recall, f1_score = evaluate_precision_recall_f1(predictions, ground_truth)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')


