from collections import Counter
import pandas as pd

def clean_text(text):
    text = text.lower()
    text = [i for i in text if i.isalnum() or i.isspace()]
    return "".join(text)

def accuracy(context, predicted_p1, predicted_p2, actual_p1, actual_p2, method):
    acutal_ans =' '.join(context[actual_p1:actual_p2])
    predicted_ans = ' '.join(context[predicted_p1:predicted_p2])
    acutal_ans = clean_text(acutal_ans)
    predicted_ans = clean_text(predicted_ans)
#     print (acutal_ans)
#     print (predicted_ans)
    if method == 'exact_match':
        return 1 if acutal_ans == predicted_ans else 0
    else:
        pred_tok = predicted_ans.split()
        actual_tok = acutal_ans.split()
        common = Counter(pred_tok) & Counter(actual_tok)
        count_common_words = sum(common.values())
        if count_common_words == 0:
            return 0.0
        precision = 1.0 * count_common_words / len(pred_tok) if len(pred_tok) > 0 else 0.0
        recall = 1.0 * count_common_words / len(actual_tok)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


def evaluate_batch_accuracy(data, vocab, p1, p2):
    p1, p2 = p1.argmax(1), p2.argmax(1)
    context_text = [[vocab.itos[word_idx.item()] for word_idx in context_idx] for context_idx in data.context_words[0]]

    exact_accuracy = [accuracy(context_text[i],
                               p1[i].item(), p2[i].item(),
                               data.start_idx[i].item(), data.end_idx[i].item(),
                               method='exact_match')
                      for i in range(data.batch_size)]

    f1_accuracy = [accuracy(context_text[i],
                            p1[i].item(), p2[i].item(),
                            data.start_idx[i].item(), data.end_idx[i].item(),
                            method='f1')
                   for i in range(data.batch_size)]

    return exact_accuracy, f1_accuracy


def evaluate(data, model, criterion, vocab, calculate_loss=True, calculate_accuracy=True):
    model.eval()
    accuracy, loss = None, None
    total_loss, examples_count = 0.0, 0
    index, exact_accurancy, f1_accuracy = [], [], []

    for batch_data in iter(data):
        p1, p2 = model(batch_data)
        if calculate_loss:
            batch_loss = criterion(p1, batch_data.start_idx) + criterion(p2, batch_data.end_idx)
            examples_count += batch_data.batch_size
            total_loss += batch_loss.item()

        if calculate_accuracy:
            index += batch_data.id
            batch_exact_accuracy, batch_f1_accuracy = evaluate_batch_accuracy(batch_data, vocab, p1, p2)
            exact_accurancy += batch_exact_accuracy
            f1_accuracy += batch_f1_accuracy

    if calculate_accuracy:
        accuracy = pd.DataFrame({'id': index, 'Exact': exact_accurancy, 'F1': f1_accuracy})

    if calculate_loss:
        loss = total_loss / examples_count
    return accuracy, loss