import torch
import logging
from config import CFG
from pathlib import Path
from utils import cal_f1score 

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

saved_model_path = Path(CFG.saved_model_path)
try:
    saved_model_path.mkdir(parents=True , exist_ok=False)
    logger.info('Model Save Path created')
except:
    logger.info("Model Save Path created' Already exists")


def train_and_eval(dataloader, valloader, model, optimizer, batch_num, writer, device ,epoch ,lr_scheduler,warmup_scheduler ):
    model.train()
    for i , batch in enumerate(dataloader):
        batch_num += 1
        sents = batch['text'].to(device)
        entity = batch['label'].to(device)
        loss = model.loss(sents,entity)
        optimizer.zero_grad()
        loss.backward()
        #clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with warmup_scheduler.dampening():
            lr_scheduler.step()

        if i % 10 == 0:
                print('[TRAIN] Epoch: {:>3d} Batch: {:4d} Loss: {:.4f}'.format(
                    epoch, (i + 1) * CFG.batch_size, loss.item()))

                writer.add_scalar('train_loss', loss.cpu().item(), batch_num)

        writer.add_scalar('loss', loss, batch_num)

    torch.save(model.state_dict(), saved_model_path/'model_epoch_{:02d}.pt'.format(epoch))

    model.eval()
    y_true, y_pred = [], []  
    val_acml_loss = 0 

    for batch in valloader:
        sents = batch['text'].to(device)
        entity = batch['label'].to(device)
        
        val_size = sents.shape[0]
        loss = model.loss(sents,entity)
        
        _, preds = model(sents,entity)
        targets = entity.cpu().detach().numpy()

        y_true.extend([ent for sen in targets for ent in sen if ent != CFG.entity_pad[1]])
        y_pred.extend([ent for sen in preds for ent in sen])

        val_acml_loss += loss.item() * val_size

    val_loss = val_acml_loss / len(valloader)
    val_f1 = cal_f1score(y_true, y_pred)
    logger.info('[VAL]   Epoch: {:>3d} Loss: {:.4f} F1-Score: {:.4f}'.format(epoch, val_loss, val_f1))
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_f1', val_f1, epoch)     
    return val_f1


    