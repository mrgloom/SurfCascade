set branch=%date:~6,4%%date:~0,2%%date:~3,2%
git stash -u
git checkout -b %branch%
git stash apply
git add .
git commit -m %branch%
git push origin %branch%
git checkout master
git stash pop
git branch -D %branch%
