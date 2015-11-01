# -*- coding: utf-8 -*-
import copy
from errno import ENOENT
import gettext as gettext_module
import os.path
import pkg_resources
import sys
__all__ = ['translation', 'gettext', 'lgettext', 'ugettext', 'ngettext', 'lngettext', 'ungettext']


# Locate a .mo file using the gettext strategy
def __find(domain, localedir='locale', languages=None, all_=False):
    """
    Return the name of a .mo file using the gettext strategy.

    :param domain: Gettext domain name (e.g. your module name)
    :param localedir: directory containing locales (give 'locale' if you have locale/fr_FR/LC_MESSAGES/domain.mo)
    :param languages: languages to find (if None: calculated with LANGUAGE, LC_ALL, LC_MESSAGES, LANG env. variables)
    :param all_: if True return a list of filenames corresponding to given languages, else return the first valid file.
    """
    # Get some reasonable defaults for arguments that were not supplied
    if languages is None:
        languages = []
        for envar in ('LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG'):
            val = os.environ.get(envar)
            if val:
                languages = val.split(':')
                break
        if 'C' not in languages:
            languages.append('C')
    # now normalize and expand the languages
    nelangs = []
    for lang in languages:
        for nelang in gettext_module._expand_lang(lang):  # pylint: disable=W0212
            if nelang not in nelangs:
                nelangs.append(nelang)
    # select a language
    result = [] if all_ else None
    for lang in nelangs:
        if lang == 'C':
            break
        mofile = '%s/%s/%s/%s.mo' % (localedir, lang, 'LC_MESSAGES', domain)
        if pkg_resources.resource_exists('document-feature-selection', mofile):
            if all_:
                result.append(mofile)
            else:
                return mofile
    return result


def translation(domain, localedir='locale', languages=None,  # pylint: disable=R0913
                class_=None, fallback=False, codeset=None):
    """

    :param domain:
    :param localedir:
    :param languages:
    :param class_:
    :param fallback:
    :param codeset:
    :return: :raise:
    """
    if class_ is None:
        class_ = gettext_module.GNUTranslations
    mofiles = __find(domain, localedir, languages, all_=True)
    if not mofiles:
        if fallback:
            return gettext_module.NullTranslations()
        raise IOError(ENOENT, 'No translation file found for domain', domain)
    # Avoid opening, reading, and parsing the .mo file after it's been done
    # once.
    result = None
    for mofile in mofiles:
        key = (class_, mofile)
        trans_obj = gettext_module._translations.get(key)  # pylint: disable=W0212
        if trans_obj is None:
            with pkg_resources.resource_stream('document-feature-selection', mofile) as fileobj:
                trans_obj = gettext_module._translations.setdefault(key, class_(fileobj))  # pylint: disable=W0212
        # Copy the translation object to allow setting fallbacks and
        # output charset. All other instance data is shared with the
        # cached object.
        trans_obj = copy.copy(trans_obj)
        if codeset:
            trans_obj.set_output_charset(codeset)
        if result is None:
            result = trans_obj
        else:
            result.add_fallback(trans_obj)
    return result


__TRANS = translation('document-feature-selection', fallback=True)
# pylint: disable=C0103
gettext = __TRANS.gettext
lgettext = __TRANS.lgettext
ngettext = __TRANS.ngettext
lngettext = __TRANS.lngettext
ugettext = __TRANS.gettext
ungettext = __TRANS.ngettext
if sys.version_info[0] == 2:
    ugettext = __TRANS.ugettext
    ungettext = __TRANS.ungettext


if __name__ == '__main__':
    import doctest
    doctest.testmod()