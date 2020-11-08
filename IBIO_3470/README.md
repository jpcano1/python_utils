# IBIO-3470
## Util Functions for Workshops
***
- Open the notebook you wish to execute
- Click where it says **Run in Google Colab**
- Click on **File >> Save a Copy in Drive**
- Execute the next code Snippet on the first cell

```python
!shred -u setup_colab.py
!wget -q "https://github.com/jpcano1/python_utils/raw/main/setup_colab_general.py" -O setup_colab_general.py
!wget -q "https://github.com/jpcano1/python_utils/raw/main/IBIO_3470/setup_colab.py" -O setup_colab.py
import setup_colab as setup
# setup.setup_project()
```