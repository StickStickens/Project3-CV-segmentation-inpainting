# third projects for CV description

## setup
ogólnie tu jest fajny samouczek gdzie masz też windowsowe comendy bo się czasem różnią
- https://doc.dvc.org/start?tab=Mac-Linux
## dvc

- Initialize DVC :
    - ```pip install dvc``` - to w setupie automatycznym jest
    - ```dvc init``` - ja to już zrobiłem i ty już nie musisz

### DVC usage
- Track model
    - ```dvc add model.pt``` # creates model.pt.dvc - the small pointer file to dvc repo. Git tracks this file not whole model file
    - ```git add model.pt.dvc .gitignore``` 
        -  ```git add model.pt.dvc```: like regular git add;
        -  ```.- gitignore``` : When you ```dvc add model.pt```, DVC automatically adds the actual model file (model.pt) to .gitignore
    - ```git commit -m "Add model checkpoint"``` # regular commit

- Track dataset (the same)
    - ```dvc add data/```
    - ```git add data.dvc .gitignore```
    - ```git commit -m "Add dataset"```

- Push data to remote storage:
    - ```dvc remote add -d myremote s3://my-bucket/path``` - to add a new “remote storage” location (lie a virtual folder);
        -  The `-d` flag means “set this remote as the default”.;
        -  `myremote` - This is just a name of remote visible in `.dvc/config` like this `['remote "myremote"'] url = s3:/my-bucket/path`;   
        - `my-bucket/path` can be:
            -  S3 bucket (s3://bucket-name/path)
            - Google Cloud Storage (gs://bucket-name/path)
            - Azure Blob (azure://container/path)
            - **we use** https://dagshub.com/ because free (i hope)
    - ```dvc push```

- When others clone the repo, they already have .dvc/ and the config in Git, so they do not run dvc init again. they run:
- ```dvc pull``` # downloads the actual datasets or model files from the DVC remote.

other commands:
- ```dvc list https://github.com/user/repo``` - Lists files in a DVC repo without cloning it 
- ```dvc status```- Shows tracked files and whether they are up-to-date locally
- ```dvc list .``` - shows local dvc fiels

git push
- `git push origin main`

there is option to store everything locally in some folder but i decided to store it publically so both of us can access data
**When to put a dataset in DVC:**
- The dataset is large (hundreds of MBs to GBs).
- You expect updates or new versions in the future.
- You want reproducibility, i.e., experiments tied to a specific dataset version.
- You want to share the dataset with the team without bloating Git. (and that is the reason i will put our dataset in DVC)

**When you can skip DVC**:
- Dataset is small (fits comfortably in Git, <50 MB).
- Dataset is static and will not change.
- You don’t mind Git tracking it directly.



+---------------------------------------------------------------------+
|        DVC has enabled anonymous aggregate usage analytics.         |
|     Read the analytics documentation (and how to opt-out) here:     |
|             <https://dvc.org/doc/user-guide/analytics>              |
|                    bendom nas śledzić                               |
+---------------------------------------------------------------------+
- Check out the documentation: <https://dvc.org/doc>
- Get help and share ideas: <https://dvc.org/chat>


## segmantation

## inpainting
