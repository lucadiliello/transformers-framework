# Test Scripts

Adjust tests to match your hardware configuration before running them.

## Run all tests

```bash
for test in tests/scripts/*; do

    if bash $test; then
        echo "Test $test passed"
    else
        echo "Test $test failed"
        break
    fi
done
```
