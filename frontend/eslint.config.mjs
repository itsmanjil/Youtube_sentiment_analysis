import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';

export default [
  // public/ ships the vendored Material Dashboard template (Bootstrap,
  // Chart.js, jQuery-era plugin scripts) — third-party code we don't own
  // the style of and don't want lint findings against.
  { ignores: ['public/**', 'dist/**', 'build/**', 'coverage/**'] },
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...js.configs.recommended.rules,
      // Only the two long-standing hooks rules, not the full v7
      // "recommended" bundle — that also pulls in ~14 React-Compiler-
      // readiness rules (static-components, purity, immutability,
      // set-state-in-effect, ...) that would require restructuring this
      // app's fetch-on-mount data-loading pattern across every page, well
      // beyond the scope of adding a lint gate.
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // This codebase deliberately swallows some errors with an empty
      // catch (e.g. AuthContext.jsx's login/refresh flows already set
      // error state and redirect regardless of the failure cause) —
      // allow that pattern rather than flagging it.
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
  },
  {
    files: ['**/*.test.{js,jsx}', 'src/setupTests.js'],
    languageOptions: {
      globals: {
        ...globals.vitest,
      },
    },
  },
];
