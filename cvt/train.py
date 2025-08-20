import torch
import torch.optim as optim
import os
import tqdm
class Train_model():
    def __init__(self,model,train_loader,test_loader,criterion,optimizer,device,  epochs,scheduler,model_save_path):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler
        self.model_save_path = model_save_path

    def train_epoch(self,model,train_loader,criterion,optimizer,device):
        model.train()
        running_loss = 0.0
        correct = 0 
        total = 0

        for batch_idx, (data, tragets) in enumerate(tqdm.tqdm(train_loader, desc="Training", leave=False)):
            data, tragets = data.to(device), tragets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, tragets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,prediction = outputs.max(1)
            total += tragets.size(0)
            correct += prediction.eq(tragets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)} , Loss: {loss.item():.4f}, Accuracy: {100. * correct / total:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc
    
    def save_checkpoint(self,model, optimizer, epoch, train_loss, val_loss, val_acc, save_path):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved: {save_path}')
    
    def validation(self):
        self.model.eval()
        running_loss =0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _,prediction = outputs.max(1)
                total += targets.size(0)
                correct +=prediction.eq(targets).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def train(self):
        best_val_acc = 0.0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            train_loss, train_acc = self.train_epoch(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
            val_loss, val_acc = self.validation()
            self.scheduler.step()
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
            print(f"learning rate: {self.optimizer.param_groups[0]['lr']:.8f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(self.model, self.optimizer, epoch+1, train_loss, val_loss, val_acc,os.path.join(self.model_save_path, 'best_model.pth'))

            if (epoch + 1)/10 == 0:
                self.save_checkpoint(self.model, self.optimizer, epoch+1, train_loss, val_loss, val_acc, os.path.join(self.model_save_path, f'checkpoint_epoch_{epoch+1}.pth'))
