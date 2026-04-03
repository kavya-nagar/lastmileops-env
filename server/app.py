[FAIL] repo: Not ready for multi-mode deployment

Issues found:
  - Missing uv.lock - run 'uv lock' to generate it
  - Missing [project.scripts] server entry point
  - server/app.py missing main() function
  - server/app.py main() function not callable (missing if __name__ == '__main__')