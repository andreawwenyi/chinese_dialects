
import torch 

def eval_ppl(model, encodings):  
    """
    source: https://huggingface.co/docs/transformers/perplexity
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nll_sum = 0
    n_tokens = 0
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        print(trg_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        
        # v2
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)

    return nll_sum.item(), ppl.item()

    ## v1
    #     nlls.append(neg_log_likelihood)

    #     prev_end_loc = end_loc
    #     if end_loc == seq_len:
    #         break
    # ppl = torch.exp(torch.stack(nlls).mean())
    # return ppl