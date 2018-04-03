def plot_time_basics_categories(fig0, obs, mod, j, rect1, rect2, rect3, rect, ref_times):
    # organize the data for taylor gram and plot
    [h_obs, d_obs, m_obs, y_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs] = obs
    [h_mod, d_mod, m_mod, y_mod, h_t_mod, d_t_mod, m_t_mod, y_t_mod] = mod

    data1 = h_obs[j, :][~h_obs[j, :].mask]
    data2 = d_obs[j, :][~d_obs[j, :].mask]
    data3 = m_obs[j, :][~m_obs[j, :].mask]

    models1, models2, models3 = [], [], []
    h_m, d_m, m_m, h_m_s, d_m_s, m_m_s = None, None, None, None, None, None
    for i in range(len(d_mod)):
        models1.append(h_mod[i][j, :][~h_obs[j, :].mask])
        models2.append(d_mod[i][j, :][~d_obs[j, :].mask])
        models3.append(m_mod[i][j, :][~m_obs[j, :].mask])

    fig0, samples1, samples2, samples3 = plot_daylor_graph_time_basic(data1, data2, data3, models1, models2, models3, fig0, rect=rect, ref_times=ref_times, bbox_to_anchor=(0.9, 0.45))


    ax0 = fig0.add_subplot(rect1)
    ax1 = fig0.add_subplot(rect2)
    ax2 = fig0.add_subplot(rect3)


    if len(data1) > 0:
        cm = plt.cm.get_cmap('RdYlBu')
        h_y = (max(np.max(data1), h_m) * 1.1 * np.ones(len(h_obs[j, :])))
        ax0.scatter(h_t_obs, h_y, c=h_obs[j, :].mask, marker='s', cmap=cm, s=1)
        d_y = (max(np.max(data2), d_m) * 1.1 * np.ones(len(d_obs[j, :])))
        ax1.scatter(d_t_obs, d_y, c=d_obs[j, :].mask, marker='s', cmap=cm, s=1)
        m_y = (max(np.max(data3), m_m) * 1.1 * np.ones(len(m_obs[j, :])))
        ax2.scatter(m_t_obs, m_y, c=m_obs[j, :].mask, marker='s', cmap=cm, s=1)
        ax0.set_ylim(min_none(np.min(data1), h_m_s), max_none(np.max(data1), h_m) * 1.15)
        ax1.set_ylim(min_none(np.min(data2), d_m_s), max_none(np.max(data2), d_m) * 1.15)
        ax2.set_ylim(min_none(np.min(data3), m_m_s), max_none(np.max(data3), m_m) * 1.15)

    else:
        # cm = plt.cm.get_cmap('RdYlBu')
        h_y = (1 * np.ones(len(h_t_obs)))
        ax0.scatter(h_t_obs, h_y, c=h_obs[j, :].mask, marker='s', cmap='Blues', s=1)
        d_y = (1* np.ones(len(d_t_obs)))
        ax1.scatter(d_t_obs, d_y, c=d_obs[j, :].mask, marker='s', cmap='Blues', s=1)
        m_y = (1 * np.ones(len(m_t_obs)))
        ax2.scatter(m_t_obs, m_y, c=m_obs[j, :].mask, marker='s', cmap='Blues', s=1)


    h_t_obs, d_t_obs, m_t_obs = h_t_obs[~h_obs[j, :].mask], d_t_obs[~d_obs[j, :].mask], m_t_obs[~m_obs[j, :].mask]

    ax0.plot(h_t_obs, data1, 'k-', label='Observed')
    ax1.plot(d_t_obs, data2, 'k-', label='Observed')
    ax2.plot(m_t_obs, data3, 'k-', label='Observed')

    for i in range(len(h_mod)):
        ax0.plot(h_t_obs, models1[i], '-', label= "Model " + str(i + 1), color=col[i])
        ax1.plot(d_t_obs, models2[i], '-', label= "Model " + str(i + 1), color=col[i])
        ax2.plot(m_t_obs, models3[i], '-', label= "Model " + str(i + 1), color=col[i])

    return fig0, ax0, ax1, ax2, [samples1, samples2, samples3]
