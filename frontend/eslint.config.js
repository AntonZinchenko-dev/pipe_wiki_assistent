import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'
import reactHooks from 'eslint-plugin-react-hooks'
import boundaries from 'eslint-plugin-boundaries'

// Направление импортов между слоями FSD проверяется отдельным правилом, а не
// договорённостью: договорённость держится до первого дедлайна.
const layers = ['app', 'pages', 'widgets', 'features', 'entities', 'shared']

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: { ecmaVersion: 2022, globals: globals.browser },
    plugins: { 'react-hooks': reactHooks, boundaries },
    settings: {
      'boundaries/elements': layers.map((layer) => ({
        type: layer,
        pattern: `src/${layer}/*`,
        mode: 'folder',
      })),
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/consistent-type-imports': 'error',
      'boundaries/element-types': [
        'error',
        {
          default: 'disallow',
          rules: layers.map((layer, index) => ({
            from: layer,
            // Импорт строго вниз по слоям: слой видит себя и всё, что ниже.
            allow: layers.slice(index),
          })),
        },
      ],
    },
  },
)
