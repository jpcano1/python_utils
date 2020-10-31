# ISIS-4825
## Util Functions for Workshops
***
- Open the notebook you wish to execute
- Click where it says **Run in Google Colab**
- Click on **File >> Save a Copy in Drive**
- Execute the next code Snippet on the first cell

```python
!shred -u setup_colab.py
!shred -u setup_colab_general.py
!wget -q "https://github.com/jpcano1/python_utils/raw/main/setup_colab_general.py" -O setup_colab_general.py
!wget -q "https://github.com/jpcano1/python_utils/raw/main/ISIS_4825/setup_colab.py" -O setup_colab.py
import setup_colab as setup
# setup.setup_workshop_8()
# setup.setup_workshop_9()
# setup.setup_workshop_10()
# setup.setup_workshop_11()
# setup.setup_workshop_12()
# setup.setup_workshop_13()
```