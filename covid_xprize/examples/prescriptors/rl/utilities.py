IPS = ['C1_School closing',
       'C2_Workplace closing',
       'C3_Cancel public events',
       'C4_Restrictions on gatherings',
       'C5_Close public transport',
       'C6_Stay at home requirements',
       'C7_Restrictions on internal movement',
       'C8_International travel controls',
       'H1_Public information campaigns',
       'H2_Testing policy',
       'H3_Contact tracing',
       'H6_Facial Coverings']

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}

# IPS_CONDENSED = ['C1,C2',
#                  'C4',
#                  'C3,C5,C7',
#                  'C6',
#                  'C8',
#                  'H1',
#                  'H2',
#                  'H3',
#                  'H6']

# These are not the true IP max values! 
# IP_MAX_VALUES = {
#     'C1,C2': 2,  # (0, 1) -> 0, (2) -> 1, (3) -> 2
#     'C4': 2,  # (0, 1) -> 0, (2, 3) -> 1, (4) -> 2
#     'C3,C5,C7': 2,
#     'C6': 2,  # (0, 1) -> 0, (2) -> 1, (3) -> 2
#     'C8': 2,  # (0, 1) -> 0, (2, 3) -> 1, (4) -> 2
#     'H1': 0,  # temporarily adjusted
#     'H2': 1,  # (0, 1) -> 0, (2, 3) -> 1
#     'H3': 1,  # (0, 1) -> 0, (2) -> 1
#     'H6': 2  # (0, 1) -> 0, (2, 3) -> 1, (4) -> 2
# }  # Groupings get us down to 2916 actions


def expand_IP(df):
    # Legacy function: should be used if we continue to use the modified 
    # IP_MAX_VALUES above

    # Map values back--take the lower of the paranthetical numbers
    df.loc[df['C1,C2'] == 2, 'C1,C2'] = 3
    df.loc[df['C1,C2'] == 1, 'C1,C2'] = 2
    df.loc[df['C4'] == 2, 'C4'] = 4
    df.loc[df['C4'] == 1, 'C4'] = 2
    df.loc[df['6'] == 2, '6'] = 3
    df.loc[df['6'] == 1, '6'] = 2
    df.loc[df['C8'] == 2, 'C8'] = 4
    df.loc[df['C8'] == 1, 'C8'] = 2
    df.loc[df['H6'] == 2, 'H6'] = 4
    df.loc[df['H6'] == 1, 'H6'] = 2
    df.loc[df['H2'] == 1, 'H2'] = 2
    df.loc[df['H3'] == 1, 'H3'] = 2

    # Change column names
    df['C2'] = df['C1,C2']
    df['C5'] = df['C3,C5,C7']
    df['C7'] = df['C3,C5,C7']
    names = ['C1,C2', 'C2', 'C3,C5,C7', 'C4', 'C5', 'C6', 'C7', 'C8', 'H1', 'H2', 'H3', 'H6']
    names = {names[i]: IPS[i] for i in range(len(names))}
    return df.rename(columns=names)
