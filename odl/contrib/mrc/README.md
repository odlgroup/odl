# MRC

- Data I/O functionality for the [MRC2014](http://www.ccpem.ac.uk/mrc_format/mrc2014.php) file format [Che+2015] including [FEI extended headers](http://www.2dx.unibas.ch/documentation/mrc-software/fei-extended-mrc-format-not-used-by-2dx),
- Support for related uncompressed binary file formats with fixed-size header.

## Example usage

```python
from odl.contrib import mrc

file_path = '/path/to/data.mrc'

with mrc.FileReaderMRC(file_path) as reader:
    header, data = reader.read()

    print('Data shape: ', reader.data_shape)
    print('Data dtype: ', reader.data_dtype)
    print('Data axis ordering: ', reader.data_axis_order)
    print('Header size (bytes): ', reader.header_size)
```

## References

[Che+2015] Cheng, A et al. *MRC2014: Extensions to the MRC format header
for electron cryo-microscopy and tomography*. Journal of Structural
Biology, 129 (2015), pp 146--150.
