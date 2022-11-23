import torch
from torch.utils.data import DataLoader
import functools


def batch_or_dataloader(agg_func=torch.cat):
    def decorator(batch_fn):
        """
        Decorator for methods in which the first arg (after `self`) can either be
        a batch or a dataloader.

        The method should be coded for batch inputs. When called, the decorator will automatically
        determine whether the first input is a batch or dataloader and apply the method accordingly.
        """
        @functools.wraps(batch_fn)
        def batch_fn_wrapper(ref, batch_or_dataloader, **kwargs):
            if isinstance(batch_or_dataloader, DataLoader): # Input is a dataloader
                list_out = [batch_fn(ref, batch.to(ref.device), **kwargs)
                            for batch, _, _ in batch_or_dataloader]

                if list_out and type(list_out[0]) in (list, tuple):
                    # Each member of list_out is a tuple/list; re-zip them and output a tuple
                    return tuple(agg_func(out) for out in zip(*list_out))
                else:
                    # Output is not a tuple
                    return agg_func(list_out)

            else: # Input is a batch
                return batch_fn(ref, batch_or_dataloader, **kwargs)

        return batch_fn_wrapper

    return decorator


def tweedie_denoising(fn):
    def wrapper(*args):
        de = args[0]
        if de.denoising_sigma is None or not de.use_tweedie_if_denoising:
            return fn(*args)
        else:
            is_training = de.training
            de.eval()
            for p in de.parameters():
                p.requires_grad = False
            had_gradients_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            samples = fn(*args)
            samples.requires_grad = True
            log_p = de.log_prob(samples)
            log_p.sum().backward()
            grad = samples.grad.detach()
            with torch.no_grad():
                samples = samples + de.denoising_sigma**2 * grad
            for p in de.parameters():
                p.requires_grad = True
            de.train(is_training)
            torch.set_grad_enabled(had_gradients_enabled)
            return samples

    return wrapper
