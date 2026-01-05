# third projects for CV description

after cloning the repo and going to repo folder do (optionally activating virtual env) (jak to odpaliłem to wywa):
- On Linux/macOS: bash setup.sh
- On Windows: double-click setup.bat or run in Command Prompt


## setup
ogólnie tu jest fajny samouczek gdzie masz też windowsowe comendy bo się czasem różnią, ale tu setupu nie ma, ale może akurat te same komendy są, chyba że na WSLu chcesz to robić żeby te same komendy linuxowe były
- https://doc.dvc.org/start?tab=Mac-Linux

Add remote for dataset storage

    `dvc remote add -d origin https://dagshub.com/StickStickens/Project3-CV-segmentation-inpainting.dvc`

Configure authentication (local only, does not go into Git)`

    dvc remote modify --local origin auth basic
    dvc remote modify --local origin user StickStickens
    dvc remote modify --local origin password <YOUR_DAGSHUB_default_TOKEN>

Add DagsHub remote (if not already present)

    git remote add dagshub https://dagshub.com/StickStickens/Project3-CV-segmentation-inpainting.git

Add GitHub remote (if not already present)

    git remote add origin https://github.com/StickStickens/Project3-CV-segmentation-inpainting.git

check:

    git remote -v    
        output:
        dagshub https://dagshub.com/StickStickens/Project3-CV-segmentation-inpainting.git (fetch)
        dagshub https://dagshub.com/StickStickens/Project3-CV-segmentation-inpainting.git (push)
        origin  https://github.com/StickStickens/Project3-CV-segmentation-inpainting (fetch)
        origin  https://github.com/StickStickens/Project3-CV-segmentation-inpainting (push)
## dvc

- Initialize DVC :
    - ```pip install dvc``` - to w setupie automatycznym jest (nie wiem czy potrzebne)
    - ```pip install "dvc[s3]"``` i to też
    - ```dvc init``` - ja to już zrobiłem i ty już nie musisz

### DVC usage
(pamiętam - https używaj bo przez protokół s3 straciłe kilka godzin życia)
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

- `git push origin main` to push to github


#### w skrócie nowe komendy do używania:
    
    dvc add data/
    git add data.dvc .gitignore
    git commit -m "Update dataset"
    dvc push


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



#### in case of problems (i dont know if safe):
dvc reset:
`# Remove old DVC config and cache`
`rm -rf .dvc`
`rm -f .dvc/config .dvc/config.local`
`rm -rf .dvc/cache`
`rm -rf .dvc/tmp`

`# Remove old .dvc files if they exist`
`find . -name "*.dvc" -exec rm -f {} \;`



## segmantation
dataset:
- https://ade20k.csail.mit.edu/

## inpainting









