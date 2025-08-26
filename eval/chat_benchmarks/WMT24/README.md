# WMT24 Machine Translation Benchmark

This benchmark evaluates language models on machine translation tasks using the WMT24++ dataset, which expands the language coverage of WMT24 to 55 languages and dialects.

## Overview

The WMT24 benchmark tests a model's ability to translate text from English to various target languages. It uses the WMT24++ dataset which includes human translations and post-edits across multiple domains, and evaluates translations using BLEU scores with domain-specific analysis.

## Supported Language Pairs

The benchmark supports 55 English-to-X language pairs:

**European Languages:**
- `en-bg_BG`: English to Bulgarian
- `en-cs_CZ`: English to Czech  
- `en-da_DK`: English to Danish
- `en-de_DE`: English to German
- `en-el_GR`: English to Greek
- `en-et_EE`: English to Estonian
- `en-fi_FI`: English to Finnish
- `en-fr_FR`: English to French (France)
- `en-fr_CA`: English to French (Canada)
- `en-hr_HR`: English to Croatian
- `en-hu_HU`: English to Hungarian
- `en-is_IS`: English to Icelandic
- `en-it_IT`: English to Italian
- `en-lt_LT`: English to Lithuanian
- `en-lv_LV`: English to Latvian
- `en-nl_NL`: English to Dutch
- `en-no_NO`: English to Norwegian
- `en-pl_PL`: English to Polish
- `en-pt_PT`: English to Portuguese (Portugal)
- `en-pt_BR`: English to Portuguese (Brazil)
- `en-ro_RO`: English to Romanian
- `en-ru_RU`: English to Russian
- `en-sk_SK`: English to Slovak
- `en-sl_SI`: English to Slovenian
- `en-sr_RS`: English to Serbian
- `en-sv_SE`: English to Swedish
- `en-uk_UA`: English to Ukrainian

**Asian Languages:**
- `en-zh_CN`: English to Mandarin (China)
- `en-zh_TW`: English to Mandarin (Taiwan)
- `en-ja_JP`: English to Japanese
- `en-ko_KR`: English to Korean
- `en-th_TH`: English to Thai
- `en-vi_VN`: English to Vietnamese
- `en-id_ID`: English to Indonesian
- `en-fil_PH`: English to Filipino

**Indian Languages:**
- `en-hi_IN`: English to Hindi
- `en-bn_IN`: English to Bengali
- `en-gu_IN`: English to Gujarati
- `en-kn_IN`: English to Kannada
- `en-ml_IN`: English to Malayalam
- `en-mr_IN`: English to Marathi
- `en-pa_IN`: English to Punjabi
- `en-ta_IN`: English to Tamil
- `en-te_IN`: English to Telugu

**Middle Eastern & African Languages:**
- `en-ar_EG`: English to Arabic (Egypt)
- `en-ar_SA`: English to Arabic (Saudi Arabia)
- `en-fa_IR`: English to Farsi
- `en-he_IL`: English to Hebrew
- `en-tr_TR`: English to Turkish
- `en-ur_PK`: English to Urdu
- `en-sw_KE`: English to Swahili (Kenya)
- `en-sw_TZ`: English to Swahili (Tanzania)
- `en-zu_ZA`: English to Zulu

**Other Languages:**
- `en-ca_ES`: English to Catalan

## Dependencies

Install the required dependencies:

```bash
pip install datasets sacrebleu numpy
```

## Usage

```python
from eval.chat_benchmarks.WMT24.eval_instruct import WMT24Benchmark

# Initialize benchmark
benchmark = WMT24Benchmark(
    language_pair="en-de_DE",      # German (Germany)
    debug=False,                   # Set to True for quick testing (10 examples)
    max_examples=100,              # Limit number of examples (None for all)
    filter_bad_source=True,        # Filter out low-quality source text
    domain_filter="news"           # Optional: filter by domain
)

# Run evaluation
results = benchmark.run_benchmark(model)
```

## Configuration Options

- `language_pair`: Language pair to evaluate (default: "en-de_DE")
- `max_examples`: Maximum number of examples to evaluate (default: None for all)
- `debug`: If True, only evaluate 10 examples for quick testing (default: False)
- `seed`: Random seed for reproducibility (default: [0, 1234, 1234, 1234])
- `max_tokens`: Maximum tokens to generate (default: 1024)
- `filter_bad_source`: Filter out examples marked as bad source (default: True)
- `domain_filter`: Optional domain filter - "news", "social", "canary", "speech", "literary" (default: None)
- `logger`: Optional logger instance
- `system_instruction`: Optional system instruction for the model

## Evaluation Metrics

The benchmark reports:

- **BLEU Score**: Primary metric using sacrebleu implementation
- **Domain-specific BLEU**: BLEU scores broken down by domain (news, social, etc.)
- **Coverage**: Percentage of examples with valid translations
- **Average Lengths**: Average word counts for references and hypotheses
- **Valid Translations**: Number of successfully generated translations
- **Domain Counts**: Number of examples per domain

## Dataset Features

- **High Quality**: Uses post-edited human translations as references
- **Multi-domain**: Includes news, social media, speech, literary, and canary domains
- **Quality Filtering**: Automatically filters out low-quality source text when enabled
- **Rich Metadata**: Includes document IDs, segment IDs, and domain information

## Dataset

- **Source**: [WMT24++ Dataset on Hugging Face](https://huggingface.co/datasets/google/wmt24pp)
- **Split Used**: Train split (only split available)
- **Format**: Direct source/target pairs with metadata
- **Reference**: Uses post-edited targets as recommended by dataset authors

## Implementation Details

- Uses sacrebleu for BLEU score calculation (standard in MT evaluation)
- Handles multiple output formats from different model types
- Provides robust translation extraction from model outputs
- Supports domain-specific evaluation and filtering
- Follows the same architectural patterns as other benchmarks in the framework
- Automatically filters bad source examples to improve evaluation quality

## Citation

If you use this benchmark, please cite the WMT24++ paper:

```bibtex
@misc{deutsch2025wmt24expandinglanguagecoverage,
      title={{WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects}}, 
      author={Daniel Deutsch and Eleftheria Briakou and Isaac Caswell and Mara Finkelstein and Rebecca Galor and Juraj Juraska and Geza Kovacs and Alison Lui and Ricardo Rei and Jason Riesa and Shruti Rijhwani and Parker Riley and Elizabeth Salesky and Firas Trabelsi and Stephanie Winkler and Biao Zhang and Markus Freitag},
      year={2025},
      eprint={2502.12404},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12404}, 
}
```
