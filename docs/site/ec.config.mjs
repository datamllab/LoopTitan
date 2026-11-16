// Code blocks: the homepage's code surface (--code ground, a grey hierarchy instead of hues: keywords muted,
// functions white, punctuation dim), no window frame, and a small mono "copy" button in the corner.
import { defineEcConfig, ExpressiveCodeTheme } from '@astrojs/starlight/expressive-code';

// the homepage's code tokens (home/tokens.css: --code, --c-fg, --c-fg-2, --c-mute, --c-punc)
const c = { bg: '#111317', fg: '#e6e8ee', fg2: '#b8bdc8', mute: '#8d93a0', punc: '#5f6573', strong: '#ffffff', sel: '#262a32' };

const theme = (name, type) =>
	new ExpressiveCodeTheme({
		name,
		type,
		colors: {
			'editor.background': c.bg,
			'editor.foreground': c.fg,
			'editor.selectionBackground': c.sel,
			'scrollbarSlider.background': c.punc,
			'scrollbarSlider.hoverBackground': c.mute,
		},
		settings: [
			{ settings: { foreground: c.fg, background: c.bg } },
			{ scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: c.punc } },
			{ scope: ['string', 'string.quoted', 'markup.inline.raw'], settings: { foreground: c.fg2 } },
			{ scope: ['constant.numeric', 'constant.language', 'constant.other', 'support.constant'], settings: { foreground: c.fg } },
			{ scope: ['keyword', 'storage', 'storage.type', 'storage.modifier', 'keyword.operator.word', 'keyword.control'], settings: { foreground: c.mute } },
			{ scope: ['entity.name.function', 'support.function', 'meta.function-call.generic'], settings: { foreground: c.strong } },
			{ scope: ['entity.name.type', 'entity.name.class', 'support.class', 'support.type'], settings: { foreground: c.fg } },
			{ scope: ['variable.parameter', 'variable.other', 'entity.other.attribute-name', 'support.type.property-name', 'meta.attribute'], settings: { foreground: c.fg } },
			{ scope: ['punctuation', 'keyword.operator', 'meta.brace'], settings: { foreground: c.punc } },
			{ scope: ['variable.parameter.shell', 'constant.other.option', 'punctuation.definition.argument'], settings: { foreground: c.mute } },
		],
	});

export default defineEcConfig({
	themes: [theme('looptitan-dark', 'dark'), theme('looptitan-light', 'light')],
	themeCssSelector: (t) => `[data-theme='${t.type}']`,
	useStarlightUiThemeColors: false,
	defaultProps: { frame: 'none' },
	styleOverrides: {
		borderRadius: '0.75rem',
		borderColor: 'var(--code-edge)',
		borderWidth: '1px',
		codeFontFamily: 'var(--mono)',
		uiFontFamily: 'var(--mono)',
		codeFontSize: '0.8125rem',
		codeLineHeight: '1.7',
		codePaddingBlock: '1rem',
		codePaddingInline: '1.25rem',
		uiFontSize: '0.75rem',
		scrollbarThumbColor: c.punc,
		scrollbarThumbHoverColor: c.mute,
		frames: {
			shadowColor: 'transparent',
			frameBoxShadowCssValue: 'none',
			editorBackground: c.bg,
			inlineButtonForeground: c.mute,
			inlineButtonBackground: '#ffffff',
			inlineButtonBackgroundIdleOpacity: '0',
			inlineButtonBackgroundHoverOrFocusOpacity: '0.07',
			inlineButtonBackgroundActiveOpacity: '0.12',
			inlineButtonBorder: 'transparent',
			tooltipSuccessBackground: '#16181d',
			tooltipSuccessForeground: '#ffffff',
		},
	},
});
